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
  arma::mat Amat;

  double* get_qr_ptr();
  const double* get_qr_ptr() const;
  void init();

public:
  QR_factorization(arma::mat&&);
  QR_factorization(const arma::mat&);

  arma::mat qy(const arma::mat&, const bool transpose = false) const;
  arma::vec qy(const arma::vec&, const bool transpose = false) const;
  arma::mat R() const;
  arma::uvec pivot() const;
};

struct dqrls_res {
  arma::mat qr;
  arma::vec coefficients;
  int rank;
  arma::ivec pivot;
  arma::vec qraux;
  bool pivoted;
};

dqrls_res
  dqrls_wrap(const arma::mat &x, arma::vec &y, double tol);

#endif
