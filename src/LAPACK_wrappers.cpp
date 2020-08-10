#include "LAPACK_wrappers.h"
#include "R_BLAS_LAPACK.h"
#include <sstream>

# define LAPACK_CHECK_ILLEGAL(info_sym, meth_name)                                           \
if(info_sym < 0){                                                                            \
  std::stringstream ss;                                                                      \
  ss << "The " << -info_sym << "-th argument to " << #meth_name  << " had an illegal value"; \
  Rcpp::stop(ss.str());                                                                      \
}

void QR_base::init(){
  /* initalize */
  for(int i = 0; i < N; ++i)
    pivot_[i] = 0;

  /* compute QR */
  int info,
     lwork = get_qr_tmp_mem_size(M, N);

  R_BLAS_LAPACK::dgeqp3(
    &M, &N, get_qr_ptr(), &M, &pivot_[0], &qraux[0], wk_mem, &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dgeqp3)

    rank = std::min(M, N);
}

arma::mat QR_base::qyt(arma::mat &B) const {
  // take copy
  int NRHS = B.n_cols, K = std::min(M, N);
  if(B.n_rows != static_cast<size_t>(M))
    Rcpp::stop("Invalid `B` matrix in `QR_factorization::qy`");

  /* compute QR */
  int info,
     lwork = get_qr_tmp_mem_size(M, N);

  R_BLAS_LAPACK::dormqr(
    "L", "T", &M, &NRHS, &K, get_qr_ptr(), &M, &qraux[0],
    B.memptr(), &M, wk_mem, &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dormqr)

  return B;
}

dqrls_res
  dqrls_wrap(const arma::mat &x, arma::vec &y, double tol)
  {
    int n = x.n_rows, p = x.n_cols, ny = 1L;

    dqrls_res out {
      x, arma::vec(p), 0L, arma::ivec(p), arma::vec(p), false
    };

    int &rank = out.rank;
    arma::mat &qr = out.qr;
    arma::ivec &pivot = out.pivot;
    for(int i = 0; i < p; ++i)
      pivot[i] = i + 1L;
    arma::vec &qraux = out.qraux,
       &coefficients = out.coefficients,
               work(2L * p),
           residuals = y,
            effects = y;

    R_BLAS_LAPACK::dqrls(
      qr.memptr(), &n, &p, y.memptr(), &ny, &tol,
      coefficients.memptr(), residuals.memptr(), effects.memptr(),
      &rank, pivot.memptr(), qraux.memptr(), work.memptr());

    bool &pivoted = out.pivoted;
    for(int i = 0L; i < p; ++i){
      if(pivot[i] != i + 1L){
        pivoted = true;
        break;
      }
    }

    for(int i = rank; i < p; ++i)
      coefficients[i] = NA_REAL;

    return out;
  }


// [[Rcpp::export]]
Rcpp::List dqrls_wrap_test(const arma::mat &x, arma::vec &y, double tol){
  auto res = dqrls_wrap(x, y, tol);
  return Rcpp::List::create(
    Rcpp::Named("qr") = Rcpp::wrap(res.qr),
    Rcpp::Named("coefficients") = Rcpp::wrap(res.coefficients),
    Rcpp::Named("rank") = res.rank,
    Rcpp::Named("pivot") = Rcpp::wrap(res.pivot),
    Rcpp::Named("qraux") = Rcpp::wrap(res.qraux),
    Rcpp::Named("pivoted") = res.pivoted);
}
