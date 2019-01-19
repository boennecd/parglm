#ifndef DD_ARMA_N_RCPP
#define DD_ARMA_N_RCPP

#define _USE_MATH_DEFINES
#include <cmath>

// Don't use openMP for elementwise operations. It is automatically if -fopenmp
// is present. Seems to cause issues on some platforms
#ifndef ARMA_DONT_USE_OPENMP
  #define ARMA_DONT_USE_OPENMP 1
#endif

//#define ARMA_NO_DEBUG
// Note: This also disables the check in inv(A, B) for whether inversion is succesfull it seems http://arma.sourceforge.net/docs.html#inv
// from armadillo config.hpp
//// Uncomment the above line if you want to disable all run-time checks.
//// This will result in faster code, but you first need to make sure that your code runs correctly!
//// We strongly recommend to have the run-time checks enabled during development,
//// as this greatly aids in finding mistakes in your code, and hence speeds up development.
//// We recommend that run-time checks be disabled _only_ for the shipped version of your program.

#include <RcppArmadillo.h>

#endif
