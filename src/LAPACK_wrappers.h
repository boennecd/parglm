#ifndef LAPACK_WRAPPERS
#define LAPACK_WRAPPERS

#include "arma_n_rcpp.h"
#include <memory>

class QR_factorization {
  const int M;
  const int N;
  std::unique_ptr<double []> qr;
  int rank;
  std::unique_ptr<double []> qraux;
  std::unique_ptr<int []> pivot_;

public:
  QR_factorization(const arma::mat&);
  arma::mat qy(const arma::mat&, const bool transpose = false) const;
  arma::vec qy(const arma::vec&, const bool transpose = false) const;
  arma::mat R() const;
  arma::uvec pivot() const;
};

#endif
