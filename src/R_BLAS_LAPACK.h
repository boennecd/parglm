#ifndef R_BLAS_LAPACK_H
#define R_BLAS_LAPACK_H

namespace R_BLAS_LAPACK {
  // wrapper for dtrtrs
  void triangular_sys_solve(
      const double*, double*, const bool, const bool, const int , const int);

  void dormqr(const char* side, const char* trans,
              const int* m, const int* n, const int* k,
              const double* a, const int* lda,
              const double* tau, double* c, const int* ldc,
              double* work, const int* lwork, int* info);

  void dgeqp3(const int* m, const int* n, double* a, const int* lda,
              int* jpvt, double* tau, double* work, const int* lwork,
              int* info);

  void dqrls(double*, int*, int*, double*, int*,
             double*, double*, double*,
             double*, int*,
             int*, double*, double*);
}

#endif
