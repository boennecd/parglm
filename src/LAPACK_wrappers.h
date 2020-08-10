#ifndef LAPACK_WRAPPERS
#define LAPACK_WRAPPERS

#include "R_BLAS_LAPACK.h"
#include "arma_n_rcpp.h"
#include <memory>

inline size_t get_qr_tmp_mem_size(int const n_rows, int const n_cols){
  constexpr int const ONE(1L),
                     MONE(-1L);

  /* dgeqp3 memory */
  size_t dgeqp3_mem = 2L * n_cols + (n_cols + 1L) * R_BLAS_LAPACK::ilaenv(
    &ONE, "DGEQRF", " ", &n_rows, &n_cols, &MONE, &MONE);

  /* dormqr memory */
  int const K = std::min(n_rows, n_cols);
  size_t dormqr_mem = std::min(
    static_cast<int>(64L), R_BLAS_LAPACK::ilaenv(
        &ONE, "DORMQR", "LT", &n_rows, &n_cols, &K, &MONE));
  dormqr_mem *= std::max(static_cast<int>(1L), n_cols);
  dormqr_mem += 64L * 65L;

  return std::max(dgeqp3_mem, dormqr_mem);
}

inline size_t get_qr_mem_size(int const n_rows, int const n_cols){

  size_t const min_mem = n_cols;
  return min_mem + get_qr_tmp_mem_size(n_rows, n_cols);
}

// TODO: use this class instead
class QR_base {
  const int M;
  const int N;
  double * const qr = nullptr;
  int rank;
  double * const qraux,
         * const wk_mem;
  std::unique_ptr<int []> pivot_;
  arma::mat Amat;

  inline double* get_qr_ptr() {
    if(!qr)
      return Amat.memptr();

    return &qr[0];
  }

  inline const double* get_qr_ptr() const {
    if(!qr)
      return Amat.memptr();

    return &qr[0];
  }
  void init();

public:
  QR_base(arma::mat &A, double * const mem):
  M(A.n_rows), N(A.n_cols), qraux(mem), wk_mem(mem + N),
  pivot_(new int[N]), Amat(A.memptr(), A.n_rows, A.n_cols, false, true) {
    init();
  }

  arma::mat qyt(arma::mat&) const;
  inline arma::mat qyt(
      const arma::mat &B) const
  {
    arma::mat out = B; /* copy */
    return qyt(std::move(out));
  }
  inline arma::vec qyt(
      const arma::vec &B) const {
    arma::mat out = B;
    return qyt(out);
  }

  inline arma::mat R() const {
    // TODO: copy that can be avoided
    return arma::trimatu(Amat.rows(0, std::min(M, N) - 1L));
  }

  arma::uvec pivot() const {
    arma::uvec out(N);
    for(int i = 0; i < N; ++i)
      out[i] = pivot_[i] - 1; /* want zero index */

    return out;
  }
};

struct QR_mem {
  std::unique_ptr<double[]> alloc_mem;

  QR_mem(size_t const M, size_t const N):
    alloc_mem(new double[get_qr_mem_size(M, N)]) { }
};

class QR_factorization final : private QR_mem, public QR_base {
public:
  QR_factorization(arma::mat &A):
  QR_mem(A.n_rows, A.n_cols), QR_base(A, alloc_mem.get()) { }
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
