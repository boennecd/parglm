#include "parallel_qr.h"
#include "arma_BLAS_LAPACK.h"

arma::mat R_F::R_rev_piv() const {
  arma::uvec piv = pivot;
  piv(piv) = arma::regspace<arma::uvec>(0, 1, piv.n_elem - 1);
  return R.cols(piv);
}

qr_parallel::worker::worker
  (std::unique_ptr<qr_data_generator> generator):
  my_generator(std::move(generator)) {}

R_F qr_parallel::worker::operator()(){
  qr_work_chunk my_chunk = my_generator->get_chunk();
  QR_factorization qr(my_chunk.X);
  arma::mat F = qr.qy(my_chunk.Y, true).rows(0, my_chunk.X.n_cols - 1);

  return R_F { qr.R(), qr.pivot(), std::move(F), my_chunk.dev };
}

qr_parallel::qr_parallel(
  ptr_vec generators, const unsigned int max_threads):
  n_threads(std::max((unsigned int)1L, max_threads)),
  pool(n_threads), futures()
  {
    while(!generators.empty()){
      submit(std::move(generators.back()));
      generators.pop_back();
    }
  }

void qr_parallel::submit(std::unique_ptr<qr_data_generator> generator){
  futures.push_back(pool.submit(worker(std::move(generator))));
}

R_F qr_parallel::compute(){
  /* gather results */
  bool is_first = true;
  arma::mat R_stack;
  arma::mat F_stack;
  arma::mat dev;
  arma::uword p = 0L, q = 0L, i = 0L;

  arma::uword num_blocks = futures.size();
  while(!futures.empty()){
    auto f = futures.begin();

    /* we assume that the first tasks are done first */
    for(arma::uword j = 0; f != futures.end() and j < n_threads; ++j, ++f){
      if(f->wait_for(std::chrono::milliseconds(10)) ==
         std::future_status::ready){
        R_F R_Fs_i = f->get();
        if(is_first){
          p = R_Fs_i.R.n_rows;
          q = R_Fs_i.F.n_rows;

          R_stack.set_size(p * num_blocks, p);
          F_stack.set_size(q * num_blocks, R_Fs_i.F.n_cols);

          dev = R_Fs_i.dev;
          is_first = false;

        } else
          dev += R_Fs_i.dev;

        R_stack.rows(i * p, (i + 1L) * p - 1L) = R_Fs_i.R_rev_piv();
        F_stack.rows(i * q, (i + 1L) * q - 1L) = std::move(R_Fs_i.F);

        ++i;
        futures.erase(f);
        break;
      }
    }
  }

  /* make new QR decomp and compute new F */
  QR_factorization qr(R_stack);
  arma::mat F = qr.qy(F_stack, true).rows(0, p - 1);

  return { qr.R(), qr.pivot(), std::move(F), dev };
}
