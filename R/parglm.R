#' @useDynLib parglm
#' @importFrom Rcpp sourceCpp
NULL

#' @export
#' @importFrom stats glm
parglm <- function(
  formula, family = gaussian, data, weights, subset,
  na.action, start = NULL, offset, control = list(...),
  contrasts = NULL, model = TRUE, x = FALSE, y = TRUE, ...){
  cl <- match.call()
  cl[[1L]] <- quote(glm)
  cl[c("method", "singular.ok")] <- list(quote(parglm.fit), FALSE)
  eval(cl, parent.frame())
}

#' @export
parglm.control <- function(
  epsilon = 1e-08, maxit = 25, trace = FALSE, nthreads = 1L,
  block_size = NULL)
{
  if (!is.numeric(epsilon) || epsilon <= 0)
    stop("value of 'epsilon' must be > 0")
  if (!is.numeric(maxit) || maxit <= 0)
    stop("maximum number of iterations must be > 0")
  stopifnot(is.numeric(nthreads) && nthreads >= 1,
            is.null(block_size) || (is.numeric(block_size) && block_size >= 1))
  list(epsilon = epsilon, maxit = maxit, trace = trace, nthreads = nthreads,
       block_size = block_size)
}

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
    use_start = use_start)

  #####
  # compute objects as in `glm.fit`
  coef <- drop(fit$coefficients)
  names(coef) <- xnames
  eta <- drop(x %*% coef) + offset
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
  rdiag <- abs(diag(fit$R))
  fit$rank <- rank <- sum(rdiag > rtol * max(ncol(x)) * .Machine$double.eps)

  if(fit$rank < ncol(x))
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
qr.R.parglmqr <- function(qr, complete = FALSE){
  qr$R
}
