context("testing results with varrious link functions agianst `glm.fit`")

to_check <- c(
  "coefficients", "residuals", "fitted.values", "rank",
  "family", "linear.predictors", "deviance", "aic", "null.deviance",
  "prior.weights", "df.residual", "df.null",
  "boundary", "formula", "terms", "data",
  "offset", "contrasts", "xlevels")

sim_func <- function(family, n, p){
  nam <- paste0(family$family, family$link)
  x_vals <- seq(-pi, pi, length.out = n)
  X <- outer(x_vals, 1:p, function(x, y) sin(x * y))
  rg <- range(rowSums(X))
  X <- X * 2 / diff(rg)

  set.seed(77311413)
  if(nam %in% c("binomiallogit", "binomialprobit", "binomialcauchit",
                "binomialcloglog")){
    inter <- 1.
    y <- family$linkinv(rowSums(X) + inter) > runif(n)

  } else if(nam %in% "binomiallog"){
    inter <- -.5
    X <- -abs(X)
    X <- X * .25 / diff(range(rowSums(X)))
    y <- family$linkinv(rowSums(X) + inter) > runif(n)

  } else if(nam %in% c("gaussianidentity", "gaussianlog")){
    inter <- 0
    y <- rnorm(n, family$linkinv(rowSums(X)), sd = .1)

  } else if(nam %in% c("gaussianinverse")){
    inter <- 1.
    X <- abs(X)
    y <- rnorm(n, family$linkinv(rowSums(X) + inter), sd = .1)

  } else if(nam %in% c("Gammainverse", "Gammaidentity", "Gammalog")){
    inter <- .5
    X <- abs(X)
    y <- rgamma(n, shape = 1, rate = 1 / family$linkinv(rowSums(X) + inter))

  }  else if(nam %in% c("poissonlog", "poissonidentity", "poissonsqrt")){
    inter <- 1.5
    X <- abs(X)
    y <- rpois(n, family$linkinv(rowSums(X) + inter))

  } else if(nam %in% c("inverse.gaussian1/mu^2", "inverse.gaussianinverse",
                       "inverse.gaussianidentity", "inverse.gaussianlog")){
    inter <- 1.5
    X <- abs(X)
    y <- SuppDists::rinvGauss(n, family$linkinv(rowSums(X) + inter), 1)

  } else stop("family not implemented")

  list(X = X, y = y, inter = inter)
}

test_expr <- expression({
  is_FAST <- method == "FAST"
  tol <- if(is_FAST) .Machine$double.eps^(1/5) else
    .Machine$double.eps^(1/4)
  expect_equal(f1[to_check], f2[to_check], label = lab,
               tolerance = tol)
  # these may differ as `glm.fit` uses the weights from the iteration prior
  # to convergence
  # expect_equal(f1$weights, f2$weights, tolerance = sqrt(1e-7), label = lab)

  s2 <- summary(f2)
  s1 <- summary(f1)

  excl <- c("call", "coefficients", "cov.unscaled", "cov.scaled",
            "dispersion", "iter")
  expect_equal(s1[!names(s1) %in% excl], s2[!names(s2) %in% excl],
               label = lab, tolerance = tol)
  na <- rownames(s1$coefficients)
  expect_equal(s1$coefficients[na, 1:2], s2$coefficients[na, 1:2],
               label = lab, tolerance = tol)

  # may also differ as the weights are not computed at the final estimates
  expect_equal(s1$dispersion, s2$dispersion, label = lab,
               tolerance = tol)
})

test_that("works with different families", {
  skip_if_not_installed("SuppDists")
  n <- 500L
  p <- 2L
  for(method in c("LAPACK", "LINPACK", "FAST"))
  for(fa in list(
    binomial("logit"), binomial("probit"), binomial("cauchit"),
    binomial("cloglog"),

    gaussian("identity"), gaussian("inverse"),
    gaussian("log"),

    Gamma("log"),

    poisson("log"), poisson("sqrt")))
  {
    tmp <- sim_func(fa, n, p)
    df <- data.frame(y = tmp$y, tmp$X)

    #####
    # no weights, no offset
    lab <- paste0(fa$family, "_", fa$link, "_", method)
    frm <- y ~ X1 + X2
    glm_control <- list(maxit = 25L, epsilon = .Machine$double.xmin)
    parglm_control <- parglm.control(
      nthreads = 2, method = method, maxit = 25L,
      epsilon = .Machine$double.xmin)
    suppressWarnings({
      f2 <- glm(frm, family = fa, data = df, control = glm_control)
      f1 <- parglm(frm, family = fa, data = df,
                   control = parglm_control)
    })

    eval(test_expr)

    #####
    # no weights, offset
    lab <- paste0(fa$family, "_", fa$link, "_", method, " w/ offset")
    offs <- seq(0, .05, length.out = n)
    suppressWarnings({
      f2 <- glm(frm, family = fa, offset = offs, data = df,
                control = glm_control)
      f1 <- parglm(frm, family = fa, offset = offs, data = df,
                   control = parglm_control)
    })
    eval(test_expr)

    #####
    # weights, no offset
    lab <- paste0(fa$family, "_", fa$link, "_", method, " w/ weigths")
    df$w <- seq(.5, 1.5, length.out = n)
    suppressWarnings({
      f2 <- glm(frm, family = fa, weights = w, data = df,
                control = glm_control)
      f1 <- parglm(frm, family = fa, weights = w, data = df,
                   control = parglm_control)
    })
    eval(test_expr)
  }
})


test_that("works with different families w/ starting values", {
  skip_if_not_installed("SuppDists")
  n <- 500L
  p <- 2L
  set.seed(77311413)
  for(method in c("LAPACK", "LINPACK", "FAST"))
  for(fa in list(
    binomial("logit"), binomial("probit"), binomial("cauchit"),
    binomial("cloglog"),

    gaussian("identity"), gaussian("inverse"),
    gaussian("log"),

    Gamma("log"),

    poisson("log"), poisson("sqrt"), poisson("identity"),

    inverse.gaussian("1/mu^2"), inverse.gaussian("inverse"),
    inverse.gaussian("identity")))
  {
    tmp <- sim_func(fa, n, p)
    df <- data.frame(y = tmp$y, tmp$X)
    df$INTER <- 1. # hack to avoid issues with NULL fit

    #####
    # no weights, no offset
    lab <- paste0(fa$family, "_", fa$link, "_", method)
    sta <- rep(1, p + 1L)
    sta[1] <- tmp$inter
    frm <- y ~ X1 + X2 + INTER - 1
    glm_control <- list(maxit = 25L, epsilon = .Machine$double.xmin)
    parglm_control <- parglm.control(
      nthreads = 2, method = method, maxit = 25L,
      epsilon = .Machine$double.xmin)
    suppressWarnings({
      f2 <- glm(frm, family = fa, start = sta, data = df,
                control = glm_control)
      f1 <- parglm(frm, family = fa, data = df,
                   control = parglm_control, start = sta)
    })
    eval(test_expr)

    #####
    # no weights, offset
    lab <- paste0(fa$family, "_", fa$link, "_", method, " w/ offset")
    sta <- rep(1, p)
    sta[1] <- tmp$inter
    frm_off <- update(frm, . ~ . - X1)
    suppressWarnings({
      f2 <- glm(frm_off, family = fa, offset = X1, start = sta, data = df,
                control = glm_control)
      f1 <- parglm(frm_off, family = fa, offset = X1, data = df,
                   control = parglm_control, start = sta)
    })
    eval(test_expr)

    #####
    # weights, no offset
    lab <- paste0(fa$family, "_", fa$link, "_", method, " w/ weigths")
    w <- runif(n)
    df$w <- n * w / sum(w)
    sta <- rep(1, p + 1L)
    sta[1] <- tmp$inter
    suppressWarnings({
      f2 <- glm(frm, family = fa, weights = w, start = sta, data = df,
                control = glm_control)
      f1 <- parglm(frm, family = fa, weights = w, start = sta, data = df,
                   control = parglm_control)
    })

    eval(test_expr)
  }
})

test_that("'method' equal to 'LINPACK' behaves as 'glm'", {
  set.seed(73640893)
  n <- 500
  p <- 5
  X <- matrix(nrow = n, ncol = p)
  for(i in 1:p)
    X[, i] <- rnorm(n, sd = sqrt(p - i + 1L))
  y <- rnorm(n) + rowSums(X)

  glm_control <- list(maxit = 25L, epsilon = .Machine$double.xmin)
  parglm_control <- parglm.control(
    nthreads = 2, maxit = 25L,
    epsilon = .Machine$double.xmin, method = "LINPACK")
  f1 <- glm(y ~ X, control = glm_control)
  f2 <- parglm(y ~ X, control = parglm_control)

  expect_equal(f1[to_check], f2[to_check])

  s1 <- summary(f1)
  s2 <- summary(f2)

  excl <- c("call", "coefficients", "cov.unscaled", "cov.scaled",
            "dispersion", "iter")
  expect_equal(s1[!names(s1) %in% excl], s2[!names(s2) %in% excl])
  expect_equal(s1$coefficients, s2$coefficients)

  # may also differ as the weights are not computed at the final estimates
  expect_equal(s1$dispersion, s2$dispersion)
})

test_that("'FASTs' fail when design matrix is singular", {
  set.seed(73640893)
  n <- 1000
  p <- 5
  X <- matrix(nrow = n, ncol = p)
  for(i in 1:p)
    X[, i] <- rnorm(n, sd = sqrt(p - i + 1L))
  y <- rnorm(n) + rowSums(X)
  X <- cbind(X[, 1:3], X[, 3:p])
  X <- cbind(X, X)

  suppressMessages(expect_error(
    f2 <- parglm(y ~ X, control = parglm.control(method = "FAST"))))
})

test_that("'parglm' yields the same as 'glm' also when one observations is not 'good'", {
  phat <- seq(.01, .99, by = .01)
  X <- log(phat / (1 - phat)) - 2
  set.seed(47313714)
  Y <- phat > runif(length(phat))

  W <- rep(1, length(Y))
  W[1] <- 0
  glm_control <- list(maxit = 25L, epsilon = .Machine$double.xmin)
  parglm_control <- parglm.control(
    nthreads = 2, maxit = 25L, epsilon = .Machine$double.xmin)
  fit <- suppressWarnings(glm(Y ~ X, binomial(), weights = W,
                              control = glm_control))
  pfit <- parglm(Y ~ X, binomial(), weights = W,
                 control = parglm_control)
  expect_equal(fit[to_check], pfit[to_check])

  Y <- rev(Y)
  X <- rev(X)
  W <- rev(W)
  fit <- suppressWarnings(glm(Y ~ X, binomial(), weights = W,
                              control = glm_control))
  pfit <- parglm(Y ~ X, binomial(), weights = W,
                 control = parglm_control)
  expect_equal(fit[to_check], pfit[to_check])
})

test_that("'stop's when there are more variables than observations", {
  set.seed(1)
  n <- 20L
  dframe <- cbind(data.frame(y = 1:n), replicate(n, rnorm(n)))

  expect_error(
    parglm(y ~ ., gaussian(), dframe),
    "not implemented with more variables than observations", fixed = TRUE)

  # check that it works with same number of observations as variables
  dframe <- dframe[, 1:n]
  fpar <- parglm(y ~ ., gaussian(), dframe, nthreads = 2)
  fglm <-    glm(y ~ ., gaussian(), dframe)
  expect_equal(coef(fpar), coef(fglm))

  # and with almost the same number of variables as observations
  dframe <- dframe[, 1:(n - 3L)]
  fpar <- parglm(y ~ ., gaussian(), dframe, nthreads = 2)
  fglm <-    glm(y ~ ., gaussian(), dframe)
  expect_equal(coef(fpar), coef(fglm))
})
