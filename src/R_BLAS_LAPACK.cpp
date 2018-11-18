#include <Rcpp.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/Applic.h>
#include "R_BLAS_LAPACK.h"

namespace R_BLAS_LAPACK {
  void triangular_sys_solve(
      const double *A, double *B, const bool is_upper, const bool trans, const int n,
      const int nrhs){
    /*
       DTRTRS solves a triangular system of the form

       A * X = B  or  A**T * X = B,

       where A is a triangular matrix of order N, and B is an N-by-NRHS
       matrix.  A check is made to verify that A is nonsingular.
     */

    int info;
    char uplo[2] = { is_upper ? 'U' : 'L' }, tra[2] = { trans ? 'T' : 'N' };

    F77_CALL(dtrtrs)(
        uplo,
        tra,
        "N",
        &n, &nrhs,
        A, &n,
        B, &n,
        &info);

    if(info != 0){
      std::stringstream str;
      str << "Got error code '" << info << "' when using LAPACK dtrtrs";
      Rcpp::stop(str.str());
    }
  }

  void dormqr(const char* side, const char* trans,
              const int* m, const int* n, const int* k,
              const double* a, const int* lda,
              const double* tau, double* c, const int* ldc,
              double* work, const int* lwork, int* info){
    F77_CALL(dormqr)(
        side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
  }

  void dgeqp3(const int* m, const int* n, double* a, const int* lda,
              int* jpvt, double* tau, double* work, const int* lwork,
              int* info){
    F77_CALL(dgeqp3)(m, n, a, lda, jpvt, tau, work, lwork, info);
  }

  void dqrls(double *x, int *n, int *p, double *y, int *ny,
             double *tol, double *b, double *rsd,
             double *qty, int *k,
             int *jpvt, double *qraux, double *work){
    F77_CALL(dqrls)(x, n, p, y, ny,
                    tol, b, rsd,
                    qty, k,
                    jpvt, qraux, work);
  }
}
