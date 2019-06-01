#ifndef DD_ARMA_N_RCPP
#define DD_ARMA_N_RCPP

#define _USE_MATH_DEFINES
#include <cmath>

// Don't use openMP for elementwise operations. It is automatically if -fopenmp
// is present. Seems to cause issues on some platforms
#ifndef ARMA_DONT_USE_OPENMP
  #define ARMA_DONT_USE_OPENMP 1
#endif

#define ARMA_NO_DEBUG

#include <RcppArmadillo.h>

#endif
