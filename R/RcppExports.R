# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

dqrls_wrap_test <- function(x, y, tol) {
    .Call('_parglm_dqrls_wrap_test', PACKAGE = 'parglm', x, y, tol)
}

parallelglm <- function(X, Ys, family, start, weights, offsets, tol, nthreads, it_max, trace, method, block_size, use_start) {
    .Call('_parglm_parallelglm', PACKAGE = 'parglm', X, Ys, family, start, weights, offsets, tol, nthreads, it_max, trace, method, block_size, use_start)
}

