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

QR_factorization::QR_factorization(const arma::mat &A):
  M(A.n_rows), N(A.n_cols), qr(new double[M * N]),
  qraux(new double[MIN(M, N)]), pivot_(new int[N]){
  // copy A
  std::memcpy(qr.get(), A.memptr(), M * N * sizeof(double));

  /* initalize */
  for(int i = 0; i < N; ++i)
    pivot_[i] = 0;

  /* compute QR */
  int info, lwork = -1;
  double tmp;
  R_BLAS_LAPACK::dgeqp3(
    &M, &N, &qr[0], &M, &pivot_[0], &qraux[0], &tmp, &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dgeqp3)

  lwork = (int) tmp;
  std::unique_ptr<double []> dwo(new double[lwork]);
  R_BLAS_LAPACK::dgeqp3(
    &M, &N, &qr[0], &M, &pivot_[0], &qraux[0], &dwo[0], &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dgeqp3)

  rank = MIN(M, N);
}

arma::mat QR_factorization::qy(
    const arma::mat &B, const bool transpose) const {
  // take copy
  arma::mat out = B;
  int NRHS = B.n_cols, K = MIN(M, N);
  if(B.n_rows != (unsigned int)M)
    Rcpp::stop("Invalid `B` matrix in `QR_factorization::qy`");

  /* compute QR */
  int info, lwork = -1;
  double tmp;
  R_BLAS_LAPACK::dormqr(
    "L", transpose ? "T" : "N", &M, &NRHS, &K, &qr[0], &M, &qraux[0],
    out.memptr(), &M, &tmp, &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dormqr)

  lwork = (int) tmp;
  std::unique_ptr<double []> work(new double[lwork]);
  R_BLAS_LAPACK::dormqr(
    "L", transpose ? "T" : "N", &M, &NRHS, &K, &qr[0], &M, &qraux[0],
    out.memptr(), &M, &work[0], &lwork, &info);
  LAPACK_CHECK_ILLEGAL(info, dormqr)

  return out;
}

arma::vec QR_factorization::qy(
    const arma::vec &B, const bool transpose) const {
  arma::mat out = B;
  return qy(out, transpose);
}

arma::mat QR_factorization::R() const {
  arma::mat out(&qr[0], M, N);
  out = out.rows(0, MIN(M, N) - 1);

  return arma::trimatu(out);
}

arma::uvec QR_factorization::pivot() const {
  arma::uvec out(N);
  for(int i = 0; i < N; ++i)
    out[i] = pivot_[i] - 1; /* want zero index */

  return out;
}
