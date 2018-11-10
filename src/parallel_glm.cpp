#include "arma_n_rcpp.h"
#include "thread_pool.h"
#include "parallel_qr.h"
#include "family.h"

/* data holder class */
class data_holder_base {
public:
  arma::vec *beta;

  /* These are not const but should not be changed... */
  arma::mat &X;
  arma::vec &Ys;
  arma::vec &weights;
  arma::vec &offsets;

  const arma::uword max_threads, p, n;
  const glm_base &family;
  const arma::uword block_size;

  data_holder_base(
    arma::mat &X, arma::vec &Ys, arma::vec &weights, arma::vec &offsets,
    const arma::uword max_threads, const arma::uword p, const arma::uword n,
    const glm_base &family, arma::uword block_size = 10000):
    X(X), Ys(Ys), weights(weights), offsets(offsets),
    max_threads(max_threads), p(p), n(n), family(std::move(family)),
    block_size(block_size)
  {}
};

struct parallelglm_res {
  const arma::vec coefficients;
  const R_F R_F;
  const arma::uword n_iter;
  const bool conv;
};

/* Class to fit glm using QR updated in chunks */
class parallelglm_class_QR {
  using uword = arma::uword;

  class glm_qr_data_generator : public qr_data_generator {
    static constexpr double zero_eps = 1e-100;

    const uword i_start, i_end;
    data_holder_base &data;
    const bool first_it;

  public:
    glm_qr_data_generator
    (uword i_start, uword i_end, data_holder_base &data, bool first_it):
    i_start(i_start), i_end(i_end), data(data), first_it(first_it) {}

    qr_work_chunk get_chunk() const override {
      /* assign objects for later use */
      arma::span my_span(i_start, i_end);
      uword n = i_end - i_start + 1;

      arma::vec y     (data.Ys.begin()      + i_start           , n, false);
      arma::vec weight(data.weights.begin() + i_start           , n, false);
      arma::vec offset(data.offsets.begin() + i_start           , n, false);
      arma::mat X     (data.X.begin() + i_start * data.p, data.p, n);

      arma::vec eta;
      if(first_it){
        eta = arma::vec(i_end - i_start + 1);
        double *eta_i = eta.begin();
        const double *y_i = y.begin();
        const double *wt = weight.begin();
        for(uword i = 0; i < n; ++i, ++eta_i, ++wt, ++y_i)
          *eta_i = data.family.initialize(*y_i, *wt);

      } else
        eta = (data.beta->t() * X).t() + offset;

      /* compute values for QR computation */
      arma::vec mu = eta, mu_eta_val = eta;
      double *m = mu.begin(), *mev = mu_eta_val.begin();
      for(uword i = 0; i < mu.n_elem; ++i, ++m, ++mev){
        *m = data.family.linkinv(*m);
        *mev = data.family.mu_eta(*mev);
      }

      arma::uvec good = arma::find(
        (weight > 0) %
          ((-zero_eps < mu_eta_val) + (mu_eta_val < zero_eps) != 2));

      const double *mu_i = mu.begin();
      const double *wt_i = weight.begin();
      const double  *y_i = y.begin();

      double dev = 0;
      for(uword i = 0; i < n; ++i, ++mu_i, ++wt_i, ++y_i)
        dev += data.family.dev_resids(*y_i, *mu_i, *wt_i);

      mu = mu(good);
      eta = eta(good);
      mu_eta_val = mu_eta_val(good);
      arma::vec var = mu;
      for(auto v = var.begin(); v != var.end(); ++v)
        *v = data.family.variance(*v);

      /* compute X and working responses and return */
      arma::vec z = (eta - offset(good)) + (y(good) - mu) / mu_eta_val;
      arma::vec w = arma::sqrt(
        (weight(good) % arma::square(mu_eta_val)) / var);

      X = X.cols(good);
      X = X.t();
      X.each_col() %= w;
      z %= w;

      arma::mat dev_mat; dev_mat = dev; /* make 1 x 1 matrix */

      return { std::move(X), std::move(z), dev_mat};
    }
  };

public:
  static parallelglm_res compute(
      arma::mat &X, arma::vec &beta0, arma::vec &Ys,arma::vec &weights,
      arma::vec &offsets, const glm_base &family, double tol,
      int nthreads, arma::uword it_max, bool trace, arma::uword block_size = 10000){
    uword p = X.n_rows;
    uword n = X.n_cols;
    data_holder_base data(X, Ys, weights, offsets, nthreads, p, n, family,
                          block_size);

    if(p != beta0.n_elem)
      Rcpp::stop("Invalid `beta0`");
    if(n != weights.n_elem)
      Rcpp::stop("Invalid `weights`");
    if(n != offsets.n_elem)
      Rcpp::stop("Invalid `offsets`");
    if(n != Ys.n_elem)
      Rcpp::stop("Invalid `Ys`");

    arma::vec beta = beta0;
    arma::uword i;
    double dev = 0.;
    std::unique_ptr<R_F> R_f_out;
    for(i = 0; i < it_max; ++i){
      arma::vec beta_old = beta;
      data.beta = &beta;

      R_f_out.reset(new R_F(get_R_f(data, i == 0)));
      /* TODO: can maybe done smarter using that R is triangular befor
       *       permutation */
      arma::mat R = R_f_out->R_rev_piv();
      beta = arma::solve(R.t(), R.t() * R_f_out->F.col(0),
                         arma::solve_opts::no_approx);
      beta = arma::solve(R    , beta,
                         arma::solve_opts::no_approx);

      if(trace){
        Rcpp::Rcout << "it " << i << "\n"
                    << "beta_old:\t" << beta_old.t()
                    << "beta:    \t" << beta.t()
                    << "Delta norm is: "<< std::endl
                    << arma::norm(beta - beta_old, 2) << std::endl
                    << "deviance is " << dev << std::endl;
      }

      double devold = dev;
      dev = R_f_out->dev(0, 0);

      if(std::abs(dev - devold) / (1e-2 + std::abs(dev)) < tol)
        break;
    }

    return { beta, *R_f_out.get(), i, i < it_max };
  }

  static R_F get_R_f(data_holder_base &data, bool first_it){
    uword n = data.X.n_cols;

    // Compute the number of blocks to create
    uword num_blocks = (n + data.block_size - 1) / data.block_size;
    std::vector<std::unique_ptr<qr_data_generator>> generators;
    generators.reserve(num_blocks);

    // setup generators
    uword i_start = 0;
    for(uword i = 0; i < num_blocks; ++i, i_start += data.block_size)
      generators.emplace_back(
        new glm_qr_data_generator(
              i_start, std::min(n - 1, i_start + data.block_size - 1),
              data, first_it));

    return qr_parallel(std::move(generators), data.max_threads).compute();
  }
};

// [[Rcpp::export]]
Rcpp::List parallelglm(
    arma::mat &X, arma::vec &Ys, std::string family, arma::vec beta0,
    arma::vec &weights, arma::vec &offsets, double tol,
    int nthreads, int it_max, bool trace,  arma::uword block_size){

  std::unique_ptr<glm_base> fam = get_fam_obj(family);
  auto result = parallelglm_class_QR::compute(
    X, beta0, Ys, weights, offsets, *fam, tol, nthreads, it_max,
    trace, block_size);

  return Rcpp::List::create(
    Rcpp::Named("coefficients") = Rcpp::wrap(result.coefficients),

    Rcpp::Named("R")      = Rcpp::wrap(result.R_F.R),
    Rcpp::Named("pivot")  = Rcpp::wrap(result.R_F.pivot + 1L),
    Rcpp::Named("F")      = Rcpp::wrap(result.R_F.F),
    Rcpp::Named("dev")    = result.R_F.dev,

    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("conv")   = result.conv);
}
