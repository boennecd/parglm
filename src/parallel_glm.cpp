#include "arma_n_rcpp.h"
#include "thread_pool.h"
#include "parallel_qr.h"
#include "family.h"
#include "R_BLAS_LAPACK.h"
#include <memory>

/* #define PARGLM_PROF */
#ifdef PARGLM_PROF
#include <gperftools/profiler.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))

/* data holder class */
class data_holder_base {
public:
  arma::vec *beta;

  /* These are not const but should not be changed... */
  arma::mat &X;
  arma::vec &Ys;
  arma::vec &weights;
  arma::vec &offsets;
  arma::vec eta;
  arma::vec mu;
  std::unique_ptr<double[]> X_work_mem;

  const arma::uword max_threads, p, n;
  const glm_base &family;
  const arma::uword block_size;

  data_holder_base(
    arma::mat &X, arma::vec &Ys, arma::vec &weights, arma::vec &offsets,
    const arma::uword max_threads, const arma::uword p, const arma::uword n,
    const glm_base &family, arma::uword block_size = 10000):
    X(X), Ys(Ys), weights(weights), offsets(offsets), eta(Ys.n_elem),
    mu(Ys.n_elem), X_work_mem(new double[X.n_elem]),
    max_threads(max_threads), p(p), n(n), family(std::move(family)),
    block_size(block_size)
  { }
};

struct parallelglm_res {
  const arma::vec coefficients;
  const ::R_F R_F;
  const double dev;
  const arma::uword n_iter;
  const bool conv;
  const arma::uword rank;
};

void thread_to_rcout(const std::string &msg){
  static std::mutex cpp_out_m;
  {
    std::lock_guard<std::mutex> lk(cpp_out_m);
    Rcpp::Rcout << msg;
  }
}

const double double_one = 1., double_zero = 0.;
const int int_one = 1;
char char_N = 'N', char_U = 'U', char_T = 'T';

/* Class to fit glm using QR updated in chunks */
class parallelglm_class_QR {
  using uword = arma::uword;

  class glm_qr_data_generator : public qr_data_generator {
    const uword i_start, i_end;
    data_holder_base &data;
    const bool do_inner;

  public:
    glm_qr_data_generator
    (uword i_start, uword i_end, data_holder_base &data,
     const bool do_inner = false):
    i_start(i_start), i_end(i_end), data(data),
    do_inner(do_inner) {}

    qr_work_chunk get_chunk() const override {
      /* assign objects for later use */
      arma::span my_span(i_start, i_end);
      arma::uword n = i_end - i_start + 1;

      arma::vec y     (data.Ys.begin()      + i_start           , n, false);
      arma::vec weight(data.weights.begin() + i_start           , n, false);
      arma::vec offset(data.offsets.begin() + i_start           , n, false);
      arma::vec eta   (data.eta.begin()     + i_start           , n, false);
      arma::vec mu    (data.mu.begin()      + i_start           , n, false);

      /* compute values for QR computation */
      arma::vec mu_eta_val = data.family.mu_eta(eta);

      arma::uvec good = arma::find((weight > 0) % (mu_eta_val != 0.));
      const bool is_all_good = good.n_elem == n;

      arma::vec var = data.family.variance(mu);

      arma::vec z = (eta - offset) + (y - mu) / mu_eta_val,
        w = arma::sqrt(weight % arma::square(mu_eta_val) / var);

      /* ensure that bad entries has zero weight */
      if(!is_all_good){
        auto good_next = good.begin();
        auto w_i = w.begin();
        arma::uword i = 0;
        for(; i < w.n_elem and good_next != good.end(); ++i, ++w_i){
          if(i == *good_next){
            ++good_next;
            continue;

          }

          *w_i = 0.;

        }

        for(; i < w.n_elem; ++i, ++w_i)
          *w_i = 0.;
      }

      const arma::uword p = data.X.n_cols;
      arma::mat X(data.X_work_mem.get() + i_start * p, n, p, false, true);
      X = data.X.rows(i_start, i_end);
      X.each_col() %= w;
      z %= w;

      arma::mat dev_mat(1L, 1L, arma::fill::zeros); /* we compute this later */

      if(do_inner){
        /* do not need to initalize when beta is zero. We do it anyway as we
         * later perform an addition for all elements */
        int dsryk_n = X.n_cols, k = X.n_rows;
        arma::mat C(dsryk_n, dsryk_n, arma::fill::zeros);

        R_BLAS_LAPACK::dsyrk(
          &char_U /*uplo*/, &char_T /*trans*/, &dsryk_n, &k /*k*/,
          &double_one /*alpha*/, X.memptr() /*A*/, &k /*lda*/,
          &double_zero /*beta*/, C.memptr(), &dsryk_n /*LDC*/);

        return { std::move(C), X.t() * z, dev_mat };
      }

      return { std::move(X), std::move(z), dev_mat};
    }
  };

  class get_inner_worker {
    const uword i_start, i_end;
    data_holder_base &data;

  public:
    get_inner_worker
    (uword i_start, uword i_end, data_holder_base &data):
    i_start(i_start), i_end(i_end), data(data) {}

    qr_work_chunk operator()() const {
      return glm_qr_data_generator(i_start, i_end, data, true).get_chunk();
    }
  };

  class worker {
    const bool first_it;
    data_holder_base &data;
    const arma::uword i_start, i_end;

  public:
    worker(const bool first_it, data_holder_base &data,
           const arma::uword i_start, const arma::uword i_end):
    first_it(first_it), data(data), i_start(i_start), i_end(i_end) { }

    double operator()(){
      arma::span my_span(i_start, i_end);
      uword n = i_end - i_start + 1;

      arma::vec eta(data.eta.begin() + i_start                 , n, false, true);
      arma::vec mu (data.mu.begin()  + i_start                 , n, false, true);
      arma::vec y     (data.Ys.begin()      + i_start          , n, false);
      arma::vec weight(data.weights.begin() + i_start          , n, false);
      arma::vec offset(data.offsets.begin() + i_start          , n, false);

      if(first_it)
        data.family.initialize(eta, y, weight);
      else {
        /* change `NA`s to zero */
        arma::vec coef = *data.beta;
        for(auto c = coef.begin(); c != coef.end(); ++c)
          if(ISNA(*c))
            *c = 0.;

        eta = offset;

        int n_i = n, p = coef.n_elem, N = data.X.n_rows;
        R_BLAS_LAPACK::dgemv(
          &char_N, &n_i /*m*/, &p /*n*/, &double_one /*alpha*/,
          data.X.memptr() + i_start /*A*/, &N /*LDA*/,
          coef.memptr() /*X*/, &int_one /*incx*/, &double_one /*beta*/,
          eta.memptr() /*Y*/, &int_one /*incy*/);
      }

      data.family.linkinv(mu, eta);

      return data.family.dev_resids(y, mu, weight);
    }
  };

  static double set_eta_n_mu(data_holder_base &data, bool first_it,
                             qr_parallel &pool, const bool use_start){
    std::vector<std::future<double> > futures;

    uword n = data.X.n_rows, i_start = 0, i_end = 0.;
    for(; i_start < n; i_start = i_end + 1L){
      i_end = std::min(n - 1, i_start + data.block_size - 1);
      if(i_start == 0L)
        /* add extra from the residual chunk */
        i_end += n % data.block_size;
      futures.push_back(
        pool.th_pool.submit(worker(
          first_it and !use_start, data, i_start, i_end)));
    }

    double dev = 0;
    while (!futures.empty())
    {
      dev += futures.back().get();
      futures.pop_back();
    }

    return dev;
  }

  static void submit_tasks(data_holder_base &data, qr_parallel &pool){
    // setup generators
    uword n = data.X.n_rows, i_start = 0, i_end = 0.;
    for(; i_start < n; i_start = i_end + 1L){
      i_end = std::min(n - 1, i_start + data.block_size - 1);
      if(i_start == 0L)
        /* add extra from the residual chunk */
        i_end += n % data.block_size;
      pool.submit(
        std::unique_ptr<qr_data_generator>(
          new glm_qr_data_generator(i_start, i_end, data)));
    }
  }

  static R_F get_R_f(data_holder_base &data, qr_parallel &pool)
    {
      submit_tasks(data, pool);

      return pool.compute();
    }

  static qr_dqrls_res get_dqrls_res(data_holder_base &data, qr_parallel &pool,
                                 const double tol)
    {
      submit_tasks(data, pool);

      return pool.compute_dqrls(tol);
    }

  struct inner_output {
    arma::mat C;
    arma::mat c;
  };

  static inner_output get_inner(data_holder_base &data, qr_parallel &pool)
  {
    std::vector<std::future<qr_work_chunk> > futures;

    uword n = data.X.n_rows, i_start = 0, i_end = 0.;
    for(; i_start < n; i_start = i_end + 1L){
      i_end = std::min(n - 1, i_start + data.block_size - 1);
      if(i_start == 0L)
        /* add extra from the residual chunk */
        i_end += n % data.block_size;
      futures.push_back(
        pool.th_pool.submit(get_inner_worker(
            i_start, i_end, data)));
    }

    inner_output out;
    bool is_first = true;
    while(!futures.empty()){
      auto o = futures.back().get();
      futures.pop_back();

      if(is_first){
        out.C = o.X;
        out.c = o.Y;
        is_first = false;
        continue;
      }

      /* TODO: could just take the upper part */
      out.C += o.X;
      out.c += o.Y;
    }

    out.C = arma::symmatu(out.C);

    return out;
  }

public:
  static parallelglm_res compute(
      arma::mat &X, arma::vec &start, arma::vec &Ys,arma::vec &weights,
      arma::vec &offsets, const glm_base &family, double tol,
      int nthreads, arma::uword it_max, bool trace, std::string method,
      arma::uword block_size = 10000, const bool use_start = false){
    uword p = X.n_cols;
    uword n = X.n_rows;
    data_holder_base data(X, Ys, weights, offsets, nthreads, p, n, family,
                          block_size);

    if(p != start.n_elem)
      Rcpp::stop("Invalid `start`");
    if(n != weights.n_elem)
      Rcpp::stop("Invalid `weights`");
    if(n != offsets.n_elem)
      Rcpp::stop("Invalid `offsets`");
    if(n != Ys.n_elem)
      Rcpp::stop("Invalid `Ys`");

    arma::vec beta = start;
    data.beta = &beta;
    arma::uword i, rank = 0L;
    double dev = 0.;
    std::unique_ptr<R_F> R_f_out;
    qr_parallel pool(std::vector<std::unique_ptr<qr_data_generator>>(),
                     data.max_threads);
    for(i = 0; i < it_max; ++i){
      arma::vec beta_old = beta;

      if(i == 0)
        dev = set_eta_n_mu(data, true, pool, use_start);

      if(method == "LAPACK"){
        R_f_out.reset(new R_F(get_R_f(data, pool)));

        /* TODO: can maybe done smarter using that R is triangular befor
         *       permutation */
        arma::mat R = R_f_out->R_rev_piv();
        beta = arma::solve(R.t(), R.t() * R_f_out->F.col(0),
                           arma::solve_opts::no_approx);
        beta = arma::solve(R    , beta,
                           arma::solve_opts::no_approx);
        rank = beta.n_elem;

      } else if(method == "LINPACK"){
        auto o = get_dqrls_res(data, pool, MIN(1e-07, tol / 1000));
        R_f_out.reset(new R_F(std::move(o.R_F)));
        for(arma::uword i = 0; i < o.R_F.pivot.n_elem; ++i)
          beta[o.R_F.pivot[i]] = o.coefficients[i];
        rank = o.rank;

      } else if(method == "FAST"){
        auto o = get_inner(data, pool);
        arma::uvec pivot(o.C.n_cols);
        for(arma::uword j = 0; j < o.C.n_cols; ++j)
          pivot[j] = j;
        R_f_out.reset(new R_F{
            arma::chol(o.C), std::move(pivot), std::move(o.c),
            arma::mat()});
        beta = arma::solve(R_f_out->R.t(), R_f_out->F.col(0),
                           arma::solve_opts::no_approx);
        beta = arma::solve(R_f_out->R    , beta,
                           arma::solve_opts::no_approx);
        rank = beta.n_elem;

      } else
        Rcpp::stop("method '" + method + "' not implemented");

      if(trace){
        Rcpp::Rcout << "it " << i << "\n"
                    << "beta_old:\t" << beta_old.t()
                    << "beta:    \t" << beta.t()
                    << "Delta norm is: "<< std::endl
                    << arma::norm(beta - beta_old, 2) << std::endl
                    << "deviance is " << dev << std::endl;
      }

      double devold = dev;
      data.beta = &beta;
      dev = set_eta_n_mu(data, false, pool, false);

      if(std::abs(dev - devold) / (.1 + std::abs(dev)) < tol)
        break;
    }

    return { beta, *R_f_out.get(), dev,
             (arma::uword)MIN(i + 1L, it_max), i < it_max, rank };
  }
};

// [[Rcpp::export]]
Rcpp::List parallelglm(
    arma::mat &X, arma::vec &Ys, std::string family, arma::vec start,
    arma::vec &weights, arma::vec &offsets, double tol,
    int nthreads, int it_max, bool trace, std::string method,
    arma::uword block_size, const bool use_start){
#ifdef PARGLM_PROF
  std::stringstream ss;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  ss << std::put_time(&tm, "profile-%d-%m-%Y-%H-%M-%S.log");
  Rcpp::Rcout << "Saving profile output to '" << ss.str() << "'" << std::endl;
  const std::string s = ss.str();
  ProfilerStart(s.c_str());
#endif

  std::unique_ptr<glm_base> fam = get_fam_obj(family);
  auto result = parallelglm_class_QR::compute(
    X, start, Ys, weights, offsets, *fam, tol, nthreads, it_max,
    trace, method, block_size, use_start);

#ifdef PARGLM_PROF
  ProfilerStop();
#endif

  return Rcpp::List::create(
    Rcpp::Named("coefficients") = Rcpp::wrap(result.coefficients),

    Rcpp::Named("R")      = Rcpp::wrap(result.R_F.R),
    Rcpp::Named("pivot")  = Rcpp::wrap(result.R_F.pivot + 1L),
    Rcpp::Named("F")      = Rcpp::wrap(result.R_F.F),
    Rcpp::Named("dev")    = result.dev,

    Rcpp::Named("n_iter") = result.n_iter,
    Rcpp::Named("conv")   = result.conv,
    Rcpp::Named("rank")   = result.rank);
}
