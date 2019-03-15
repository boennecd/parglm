#include "LAPACK_wrappers.h"
#include "R_BLAS_LAPACK.h"
#include <sstream>

#define MIN(a,b) (((a)<(b))?(a):(b))

# define LAPACK_CHECK_ILLEGAL(info_sym, meth_name)                                           \
if(info_sym < 0){                                                                            \
  std::stringstream ss;                                                                      \
  ss << "The " << -info_sym << "-th argument to " << #meth_name  << " had an illegal value"; \
  Rcpp::stop(ss.str());                                                                      \
}

QR_factorization::QR_factorization(arma::mat &A):
  M(A.n_rows), N(A.n_cols), qraux(new double[MIN(M, N)]),
  pivot_(new int[N]), Amat(A.memptr(), A.n_rows, A.n_cols, false) {
  init();
}

double* QR_factorization::get_qr_ptr() {
  if(!qr)
    return Amat.memptr();

  return &qr[0];
}

const double* QR_factorization::get_qr_ptr() const {
  if(!qr)
    return Amat.memptr();

  return &qr[0];
}

void QR_factorization::init(){
  /* initalize */
  for(int i = 0; i < N; ++i)
    pivot_[i] = 0;

  /* compute QR */
  int info, lwork = -1;
  double tmp;
  R_BLAS_LAPACK::dgeqp3(
    &M, &N, get_qr_ptr(), &M, &pivot_[0], &qraux[0], &tmp, &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dgeqp3)

    lwork = (int) tmp;
  std::unique_ptr<double []> dwo(new double[lwork]);
  R_BLAS_LAPACK::dgeqp3(
    &M, &N, get_qr_ptr(), &M, &pivot_[0], &qraux[0], &dwo[0], &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dgeqp3)

    rank = MIN(M, N);
}

arma::mat QR_factorization::qy(
    const arma::mat &B, const bool transpose) const
{
  arma::mat out = B; /* copy */
  return qy(std::move(out), transpose);
}

arma::mat QR_factorization::qy(
    arma::mat &B, const bool transpose) const {
  // take copy
  int NRHS = B.n_cols, K = MIN(M, N);
  if(B.n_rows != (unsigned int)M)
    Rcpp::stop("Invalid `B` matrix in `QR_factorization::qy`");

  /* compute QR */
  int info, lwork = -1;
  double tmp;
  R_BLAS_LAPACK::dormqr(
    "L", transpose ? "T" : "N", &M, &NRHS, &K, get_qr_ptr(), &M, &qraux[0],
    B.memptr(), &M, &tmp, &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dormqr)

  lwork = (int) tmp;
  std::unique_ptr<double []> work(new double[lwork]);
  R_BLAS_LAPACK::dormqr(
    "L", transpose ? "T" : "N", &M, &NRHS, &K, get_qr_ptr(), &M, &qraux[0],
    B.memptr(), &M, &work[0], &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dormqr)

  return B;
}

arma::vec QR_factorization::qy(
    const arma::vec &B, const bool transpose) const {
  arma::mat out = B;
  return qy(out, transpose);
}

arma::mat QR_factorization::R() const {
  return arma::trimatu(Amat.rows(0, MIN(M, N) - 1));
}

arma::uvec QR_factorization::pivot() const {
  arma::uvec out(N);
  for(int i = 0; i < N; ++i)
    out[i] = pivot_[i] - 1; /* want zero index */

  return out;
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
    arma::vec &qraux = out.qraux, &coefficients = out.coefficients,
      work(2L * p), residuals = y, effects = y;

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
