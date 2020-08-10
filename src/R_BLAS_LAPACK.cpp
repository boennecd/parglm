#include <Rconfig.h>
#include <R_ext/BLAS.h>
#ifndef FCLEN
#define FCLEN
#endif
#ifndef FCONE
#define FCONE
#endif
#include <R_ext/Lapack.h>
#include <Rcpp.h>

extern "C"
{
#ifdef FC_LEN_T
  int F77_NAME(ilaenv)(
      int const* /* ISPEC */, char const* /* NAME */,
      char const* /* OPTS */, int const* /* N1 */,
      int const* /* N2 */, int const* /* N3 */, int const* /* N4 */,
      FC_LEN_T, FC_LEN_T);
#else
  int F77_NAME(ilaenv)(
      int const* /* ISPEC */, char const* /* NAME */,
      char const* /* OPTS */, int const* /* N1 */,
      int const* /* N2 */, int const* /* N3 */, int const* /* N4 */);
#endif
}

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
        &info FCONE FCONE FCONE);

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
        side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info
        FCONE FCONE);
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

  void dgemv(const char *trans, const int *m, const int *n,
             const double *alpha, const double *a, const int *lda,
             const double *x, const int *incx, const double *beta,
             double *y, const int *incy){
    F77_CALL(dgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy
                    FCONE);
  }

  void dsyrk(const char *uplo, const char *trans,
             const int *n, const int *k,
             const double *alpha, const double *a, const int *lda,
             const double *beta, double *c, const int *ldc){
    F77_CALL(dsyrk)(uplo, trans, n, k, alpha, a, lda, beta, c, ldc
                    FCONE FCONE);
  }

  int ilaenv(int const *ispec, std::string const &name,
             std::string const &opts,
             int const *N1, int const *N2, int const *N3, int const *N4){
#ifdef FC_LEN_T
    return F77_CALL(ilaenv)(ispec, name.c_str(), opts.c_str(), N1,
                    N2, N3, N4, name.size(), opts.size());
#else
    return F77_CALL(ilaenv)(ispec, name.c_str(), opts.c_str(), N1,
                    N2, N3, N4);
#endif
  }
}
