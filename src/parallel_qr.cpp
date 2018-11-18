#include "parallel_qr.h"
#include "LAPACK_wrappers.h"

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
  futures(), th_pool(n_threads)
  {
    while(!generators.empty()){
      submit(std::move(generators.back()));
      generators.pop_back();
    }
  }

void qr_parallel::submit(std::unique_ptr<qr_data_generator> generator){
  futures.push_back(th_pool.submit(worker(std::move(generator))));
}

qr_parallel::get_stacks_res_obj qr_parallel::get_stacks_res(){
  get_stacks_res_obj out;

  bool is_first = true;
  arma::mat &R_stack = out.R_stack;
  arma::mat &F_stack = out.F_stack;
  arma::mat &dev     = out.dev;
  arma::uword &p = out.p, q = 0L, i = 0L;
  p = 0L;

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

  return out;

}

R_F qr_parallel::compute(){
  auto stacked = get_stacks_res();

  /* make new QR decomp and compute new F */
  QR_factorization qr(stacked.R_stack);
  arma::mat F = qr.qy(stacked.F_stack, true).rows(0, stacked.p - 1);

  return { qr.R(), qr.pivot(), std::move(F), stacked.dev };
}

qr_dqrls_res qr_parallel::compute_dqrls(const double tol){
  auto stacked = get_stacks_res();

  arma::vec y = stacked.F_stack.col(0); /* TODO: do not use copy */
  auto o = dqrls_wrap(stacked.R_stack, y, tol);

  arma::uword di = std::min(o.qr.n_cols, o.qr.n_rows) - 1L;

  arma::mat R = o.qr.submat(0L, 0L, di, di);
  R = arma::trimatu(R);
  arma::uvec pivot(o.pivot.n_elem);
  for(arma::uword i = 0; i < o.pivot.n_elem; ++i)
    pivot[i] = o.pivot[i] - 1L;

  return { R_F { R, std::move(pivot), arma::mat(),
                 stacked.dev }, o.coefficients };
}
