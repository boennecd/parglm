#include "arma_n_rcpp.h"
#include "thread_pool.h"
#include "parallel_qr.h"
#include "family.h"
#include "R_BLAS_LAPACK.h"
#include <memory>
#include "constant.h"

#ifdef PARGLM_PROF
#include <gperftools/profiler.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#endif

inline size_t floor_mult(size_t const num, size_t const denom){
  size_t const tmp = num / denom;
  return std::max(tmp, static_cast<size_t>(1L)) * denom;
}

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

  const arma::uword max_threads, p, n;
  const glm_base &family;
  const arma::uword block_size;

  data_holder_base(
    arma::mat &X, arma::vec &Ys, arma::vec &weights, arma::vec &offsets,
    const arma::uword max_threads, const arma::uword p, const arma::uword n,
    const glm_base &family, arma::uword b_size = 10000):
    X(X), Ys(Ys), weights(weights), offsets(offsets), eta(Ys.n_elem),
    mu(Ys.n_elem),
    max_threads(max_threads), p(p), n(n), family(std::move(family)),
    block_size(floor_mult(b_size, cacheline_size() / sizeof(double)))
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

const double double_one = 1., double_zero = 0.;
const int int_one = 1;
char char_N = 'N', char_U = 'U', char_T = 'T';

inline void inplace_copy
  (arma::mat &X, const arma::mat &Y, const arma::uword start,
   const arma::uword end)
  {
    double *x = X.begin();
    const double *y = Y.begin() + start;

    std::size_t n_ele = end - start + 1L;
    for(unsigned int i = 0; i < X.n_cols;
        ++i, x += X.n_rows, y += Y.n_rows)
      std::memcpy(x, y, n_ele * sizeof(double));
  }

namespace {
/* handles working memory */
static size_t wk_mem_per_thread  = 0L,
              current_wk_size    = 0L;
static std::unique_ptr<double[]> current_wk_mem =
  std::unique_ptr<double[]>();

double * get_working_memory(thread_pool const &pool){
  size_t const my_num = pool.get_id();
  return current_wk_mem.get() + my_num * wk_mem_per_thread;
}

void set_working_memory
  (size_t const max_n, size_t const max_p, size_t const n_threads){
  constexpr size_t const mult = cacheline_size() / sizeof(double),
                     min_size = 2L * mult;
  size_t m_dim = 2L * max_n + max_n * max_p;
  m_dim = std::max(m_dim, min_size);
  m_dim = (m_dim + mult - 1L) / mult;
  m_dim *= mult;
  wk_mem_per_thread = m_dim;

  size_t const n_t = std::max(n_threads, static_cast<size_t>(1L)) + 1L,
          new_size = n_t * m_dim;
  if(new_size > current_wk_size){
    current_wk_mem.reset(new double[new_size]);
    current_wk_size = new_size;

  }
}

void set_working_memory(data_holder_base const &data){
  size_t const resid = data.n % data.block_size + 1L;
  set_working_memory(data.block_size + resid, data.p, data.max_threads);
}
} // namespace


/* Class to fit glm using QR updated in chunks */
class parallelglm_class_QR {
  using uword = arma::uword;

  class glm_qr_data_generator final : public qr_data_generator {
    const uword i_start, i_end;
    data_holder_base &data;
    thread_pool const & pool;
    const bool do_inner;

  public:
    glm_qr_data_generator
    (uword i_start, uword i_end, data_holder_base &data,
     thread_pool const &pool,const bool do_inner = false):
    i_start(i_start), i_end(i_end), data(data), pool(pool),
    do_inner(do_inner) {}

    qr_work_chunk get_chunk() const override {
      /* assign objects for later use */
      arma::uword const n = i_end - i_start + 1;

      double * const wk_mem = get_working_memory(pool);
      size_t wk_cur = 0L;
      auto get_wk_mem = [&](size_t const siz){
        double * out = wk_mem + wk_cur;
        wk_cur += siz;
        return out;
      };

      arma::vec y(data.Ys.begin()      + i_start, n, false, true),
           weight(data.weights.begin() + i_start, n, false, true),
           offset(data.offsets.begin() + i_start, n, false, true),
              eta(data.eta.begin()     + i_start, n, false, true),
               mu(data.mu.begin()      + i_start, n, false, true),
                w(get_wk_mem(n)                 , n, false, true),
                z(get_wk_mem(n)                 , n, false, true);

      /* compute values for QR computation */
      for(size_t i = 0; i < n; ++i){
        double const var = data.family.variance(mu[i]),
              mu_eta_val = data.family.mu_eta(eta[i]);

        z[i] = (eta[i] - offset[i]) + (y[i] - mu[i]) / mu_eta_val;
        w[i] = std::sqrt(weight[i] * mu_eta_val * mu_eta_val / var);

        bool is_good = w[i] > 0 and mu_eta_val != 0.;
        if(!is_good)
          w[i] = 0.;
      }

      const arma::uword p = data.X.n_cols;
      arma::mat X(get_wk_mem(n * p), n, p, false, true);
      inplace_copy(X, data.X, i_start, i_end);
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
    thread_pool const &pool;

  public:
    get_inner_worker
    (uword i_start, uword i_end, data_holder_base &data,
     thread_pool const &pool):
    i_start(i_start), i_end(i_end), data(data), pool(pool) {}

    qr_work_chunk operator()() const {
      return glm_qr_data_generator
        (i_start, i_end, data, pool, true).get_chunk();
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
      uword const n = i_end - i_start + 1;

      arma::vec eta(data.eta.begin() + i_start       , n, false, true);
      arma::vec mu (data.mu.begin()  + i_start       , n, false, true);
      arma::vec y     (data.Ys.begin()      + i_start, n, false, true);
      arma::vec weight(data.weights.begin() + i_start, n, false, true);
      arma::vec offset(data.offsets.begin() + i_start, n, false, true);

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

    set_working_memory(data);

    for(; i_start < n; i_start = i_end + 1L){
      i_end = std::min(n - 1, i_start + data.block_size - 1);
      bool const is_end = i_end + data.block_size > n - 1;
      if(is_end and i_end < n - 1L)
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

    set_working_memory(data);

    for(; i_start < n; i_start = i_end + 1L){
      i_end = std::min(n - 1, i_start + data.block_size - 1);
      bool const is_end = i_end + data.block_size > n - 1;
      if(is_end and i_end < n - 1L)
        /* add extra from the residual chunk */
        i_end += n % data.block_size;
      pool.submit(
        std::unique_ptr<qr_data_generator>(
          new glm_qr_data_generator(i_start, i_end, data, pool.th_pool)));
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

    set_working_memory(data);

    for(; i_start < n; i_start = i_end + 1L){
      i_end = std::min(n - 1, i_start + data.block_size - 1);
      bool const is_end = i_end + data.block_size > n - 1;
      if(is_end and i_end < n - 1L)
        /* add extra from the residual chunk */
        i_end += n % data.block_size;
      futures.push_back(
        pool.th_pool.submit(get_inner_worker(
            i_start, i_end, data, pool.th_pool)));
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
        auto o = get_dqrls_res(data, pool, std::min(1e-07, tol / 1000));
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
             std::min(static_cast<arma::uword>(i + 1L), it_max),
             i < it_max, rank };
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
