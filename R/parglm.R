#' @useDynLib parglm
#' @importFrom Rcpp sourceCpp
NULL

#' @name parglm
#' @title Fitting Generalized Linear Models in Parallel
#'
#' @description Function like \code{\link{glm}} which can make the computation
#' in parallel. The function supports most families listed in \code{\link{family}}.
#' See "\code{vignette("parglm", "parglm")}" for run time examples.
#'
#' @param formula an object of class \code{\link{formula}}.
#' @param family a \code{\link{family}} object.
#' @param data an optional data frame, list or environment containing the variables
#' in the model.
#' @param weights an optional vector of 'prior weights' to be used in the fitting process. Should
#' be \code{NULL} or a numeric vector.
#' @param subset	an optional vector specifying a subset of observations to be used in
#' the fitting process.
#' @param na.action a function which indicates what should happen when the data contain \code{NA}s.
#' @param start starting values for the parameters in the linear predictor.
#' @param etastart starting values for the linear predictor. Not supported.
#' @param mustart starting values for the vector of means. Not supported.
#' @param offset this can be used to specify an a priori known component to be
#' included in the linear predictor during fitting.
#' @param control	a list of parameters for controlling the fitting process.
#' For parglm.fit this is passed to \code{\link{parglm.control}}.
#' @param model	a logical value indicating whether model frame should be included
#' as a component of the returned value.
#' @param x,y For \code{parglm}: logical values indicating whether the response vector
#' and model matrix used in the fitting process should be returned as components of the
#' returned value.
#'
#' For \code{parglm.fit}: \code{x} is a design matrix of dimension \code{n * p}, and
#' \code{y} is a vector of observations of length \code{n}.
#' @param contrasts	an optional list. See the \code{contrasts.arg} of
#' \code{\link{model.matrix.default}}.
#' @param intercept	logical. Should an intercept be included in the null model?
#' @param ...	For \code{parglm}: arguments to be used to form the default \code{control} argument
#' if it is not supplied directly.
#'
#' For \code{parglm.fit}: unused.
#'
#' @return
#' \code{glm} object as returned by \code{\link{glm}} but differs mainly by the \code{qr}
#' element. The \code{qr} element in the object returned by \code{parglm}(\code{.fit}) only has the \eqn{R}
#' matrix from the QR decomposition.
#'
#' @examples
#' # small example from `help('glm')`. Fitting this model in parallel does
#' # not matter as the data set is small
#' clotting <- data.frame(
#'   u = c(5,10,15,20,30,40,60,80,100),
#'   lot1 = c(118,58,42,35,27,25,21,19,18),
#'   lot2 = c(69,35,26,21,18,16,13,12,12))
#' f1 <- glm   (lot1 ~ log(u), data = clotting, family = Gamma)
#' f2 <- parglm(lot1 ~ log(u), data = clotting, family = Gamma,
#'              control = parglm.control(nthreads = 2L))
#' all.equal(coef(f1), coef(f2))
#'
#' @importFrom stats glm
#' @export
parglm <- function(
  formula, family = gaussian, data, weights, subset,
  na.action, start = NULL, offset, control = list(...),
  contrasts = NULL, model = TRUE, x = FALSE, y = TRUE, ...){
  cl <- match.call()
  cl[[1L]] <- quote(glm)
  cl[c("method", "singular.ok")] <- list(quote(parglm::parglm.fit), FALSE)
  eval(cl, parent.frame())
}

#' @title Auxiliary for Controlling GLM Fitting in Parallel
#'
#' @description
#' Auxiliary function for \code{\link{parglm}} fitting.
#'
#' @param epsilon positive convergence tolerance.
#' @param maxit integer giving the maximal number of IWLS iterations.
#' @param trace logical indicating if output should be produced doing estimation.
#' @param nthreads number of cores to use. You may get the best performance by
#' using your number of physical cores if your data set is sufficiently large.
#' Using the number of physical CPUs/cores may yield the best performance
#' (check your number e.g., by calling \code{parallel::detectCores(logical = FALSE)}).
#' @param block_size number of observation to include in each parallel block.
#' @param method string specifying which method to use. Either \code{"LINPACK"},
#' \code{"LAPACK"}, or \code{"FAST"}.
#'
#' @details
#' The \code{LINPACK} method uses the same QR method as \code{\link{glm.fit}} for the final QR decomposition.
#' This is the \code{dqrdc2} method described in \code{\link[base]{qr}}. All other QR
#' decompositions but the last are made with \code{DGEQP3} from \code{LAPACK}.
#' See Wood, Goude, and Shaw (2015) for details on the QR method.
#'
#' The \code{FAST} method computes the Fisher information and then solves the normal
#' equation. This is faster but less numerically stable.
#'
#' @references
#' Wood, S.N., Goude, Y. & Shaw S. (2015) Generalized additive models for
#' large datasets. Journal of the Royal Statistical Society, Series C
#' 64(1): 139-155.
#'
#' @return
#' A list with components named as the arguments.
#'
#' @examples
#' # use one core
#'clotting <- data.frame(
#'  u = c(5,10,15,20,30,40,60,80,100),
#'  lot1 = c(118,58,42,35,27,25,21,19,18),
#'  lot2 = c(69,35,26,21,18,16,13,12,12))
#' f1 <- parglm(lot1 ~ log(u), data = clotting, family = Gamma,
#'              control = parglm.control(nthreads = 1L))
#'
#' # use two cores
#' f2 <- parglm(lot1 ~ log(u), data = clotting, family = Gamma,
#'              control = parglm.control(nthreads = 2L))
#' all.equal(coef(f1), coef(f2))
#'
#' @export
parglm.control <- function(
  epsilon = 1e-08, maxit = 25, trace = FALSE, nthreads = 1L,
  block_size = NULL, method = "LINPACK")
{
  if (!is.numeric(epsilon) || epsilon <= 0)
    stop("value of 'epsilon' must be > 0")
  if (!is.numeric(maxit) || maxit <= 0)
    stop("maximum number of iterations must be > 0")
  stopifnot(
    is.numeric(nthreads) && nthreads >= 1,
    is.null(block_size) || (is.numeric(block_size) && block_size >= 1),
    method %in% c("LAPACK", "LINPACK", "FAST"))
  list(epsilon = epsilon, maxit = maxit, trace = trace, nthreads = nthreads,
       block_size = block_size, method = method)
}

#' @rdname parglm
#' @importFrom stats gaussian binomial Gamma inverse.gaussian poisson
#' @export
parglm.fit <- function(
  x, y, weights = rep(1, nobs), start = NULL, etastart = NULL,
  mustart = NULL, offset = rep(0, nobs), family = gaussian(),
  control = list(), intercept = TRUE, ...){
  .check_fam(family)
  stopifnot(nrow(x) == length(y))

  if(!is.null(mustart))
    warning(sQuote("mustart"), " will not be used")
  if(!is.null(etastart))
    warning(sQuote("etastart"), " will not be used")

  #####
  # like in `glm.fit`
  control <- do.call("parglm.control", control)
  x <- as.matrix(x)
  xnames <- dimnames(x)[[2L]]
  ynames <- if(is.matrix(y)) rownames(y) else names(y)
  conv <- FALSE
  nobs <- NROW(y)
  nvars <- ncol(x)
  EMPTY <- nvars == 0

  if(EMPTY)
    stop("not implemented for empty model")
  if(NCOL(y) > 1L)
    stop("Multi column ", sQuote("y"), " is not supported")

  if (is.null(weights))
    weights <- rep.int(1, nobs)
  if (is.null(offset))
    offset <- rep.int(0, nobs)

  n_min_per_thread <- 10L
  n_per_thread <- nrow(x) / control$nthreads
  if(n_per_thread < n_min_per_thread){
    nthreads_new <- nrow(x) %/% n_min_per_thread
    if(nthreads_new < 1L)
      nthreads_new <- 1L

    warning(
      "Too few observation compared to the number of threads. ",
      nthreads_new, " thread(s) will be used instead of ",
      control$nthreads, ".")

    control$nthreads <- nthreads_new
  }

  block_size <- if(!is.null(control$block_size))
    control$block_size else
      if(control$nthreads > 1L)
        max(nrow(x) / control$nthreads, control$nthreads) else
          nrow(x)

  use_start <- !is.null(start)
  fit <- parallelglm(
    X = x, Ys = y, family = paste0(family$family, "_", family$link),
    start = if(use_start) start else numeric(ncol(x)), weights = weights,
    offsets = offset, tol = control$epsilon, nthreads = control$nthreads,
    it_max = control$maxit, trace = control$trace, block_size = block_size,
    use_start = use_start, method = control$method)

  #####
  # compute objects as in `glm.fit`
  coef <- drop(fit$coefficients)
  names(coef) <- xnames
  coef_dot <- ifelse(is.na(coef), 0, coef)
  eta <- drop(x %*% coef_dot) + offset
  good <- weights > 0
  mu <- family$linkinv(eta)
  mu.eta.val <- family$mu.eta(eta)
  good <- (weights > 0) & (mu.eta.val != 0)
  w <- sqrt((weights[good] * mu.eta.val[good]^2) / family$variance(mu)[good])

  wt <- rep.int(0, nobs)
  wt[good] <- w^2

  residuals <- (y - mu) / mu.eta.val

  dev <- drop(fit$dev) # should mabye re-compute...

  conv <- fit$conv
  iter <- fit$n_iter

  boundary <- FALSE # TODO: not as in `glm.fit`

  Rmat <- fit$R
  dimnames(Rmat) <- list(xnames, xnames)

  names(residuals) <- names(mu) <- names(eta) <- names(wt) <- names(weights) <-
    names(y) <- ynames

  # do as in `Matrix::rankMatrix`
  rtol <- max(dim(x)) * .Machine$double.eps
  fit$rank <- rank <- fit$rank
  rdiag <- abs(diag(fit$R))
  if(control$method != "LINPACK" && any(rdiag <= rtol * max(rdiag)))
    warning("Non-full rank problem. Output may not be reliable.")

  #####
  # do roughly as in `glm.fit`
  if (!conv)
    warning("parglm.fit: algorithm did not converge", call. = FALSE)

  wtdmu <-
    if (intercept) sum(weights * y)/sum(weights) else family$linkinv(offset)
  nulldev <- sum(family$dev.resids(y, wtdmu, weights))

  n.ok <- nobs - sum(weights==0)
  nulldf <- n.ok - as.integer(intercept)
  rank <- fit$rank
  resdf  <- n.ok - rank
  #-----------------------------------------------------------------------------
  # calculate AIC
  # we need to initialize n if the family is `binomial`. As of 11/11/2018 two
  # column ys are not allowed so this is easy
  n <- rep(1, nobs)
  aic.model <- family$aic(y, n, mu, weights, dev) + 2*rank
  #-----------------------------------------------------------------------------
  list(coefficients = coef, residuals = residuals, fitted.values = mu,
       # effects = fit$effects, # TODO: add
       R = Rmat, rank = rank,
       qr = structure(c(fit, list(qr = fit$R)), class = "parglmqr"),
       family = family,
       linear.predictors = eta, deviance = dev, aic = aic.model,
       null.deviance = nulldev, iter = iter, weights = wt,
       prior.weights = weights, df.residual = resdf, df.null = nulldf,
       y = y, converged = conv, boundary = boundary)
}


.check_fam <- function(family){
  stopifnot(
    inherits(family, "family"),
    paste(family$family, family$link) %in%
      sapply(parglm_supported(), function(x) paste(x$family, x$link)))
}

parglm_supported <- function()
  list(
    gaussian("identity"), gaussian("log"), gaussian("inverse"),

    binomial("logit"), binomial("probit"), binomial("cauchit"),
    binomial("log"), binomial("cloglog"),

    Gamma("inverse"), Gamma("identity"), Gamma("log"),

    poisson("log"), poisson("identity"), poisson("sqrt"),

    inverse.gaussian("1/mu^2"), inverse.gaussian("inverse"),
    inverse.gaussian("identity"), inverse.gaussian("log"))

#' @importFrom Matrix qr.R
#' @export
qr.R.parglmqr <- function(x, ...){
  x$R
}
