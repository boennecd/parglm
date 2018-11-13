context("testing results with varrious link functions agianst `glm.fit`")

to_check <- c(
  "coefficients", "residuals", "fitted.values", "rank",
  "family", "linear.predictors", "deviance", "aic", "null.deviance",
  "iter", "prior.weights", "df.residual", "df.null",
  "converged", "boundary", "formula", "terms", "data",
  "offset", "contrasts", "xlevels")

sim_func <- function(family, n, p){
  nam <- paste0(family$family, family$link)
  if(nam %in% c("binomiallogit", "binomialprobit", "binomialcauchit",
                "binomialcloglog")){
    sds <- p:1
    sds <- sqrt(sds^2 / sum(sds^2))
    X <- matrix(nrow = n, ncol = p)
    set.seed(77311413)
    for(i in 1:p)
      X[, i] <- rnorm(n, 1/p, sds[i])
    y <- family$linkinv(rowSums(X)) > runif(n)

  } else if(nam %in% "binomiallog"){
    c <- matrix(rnorm(n * p, mean = -1/log(p), sd = 1/sqrt(p)), n)
    fac <- rowSums(X)
    mif <- log(.0001)
    maf <- log(.9999)
    fac <- ifelse(fac < mif | fac > maf, fac, 1)
    X <- X / fac

    y <- family$linkinv(rowSums(X)) > runif(n)

  } else if(nam %in% c("gaussianidentity", "gaussianlog")){
    sds <- p:1
    sds <- sqrt(sds^2 / sum(sds^2))
    X <- matrix(nrow = n, ncol = p)
    set.seed(77311413)
    for(i in 1:p)
      X[, i] <- rnorm(n, 0, sds[i])
    y <- rnorm(n, family$linkinv(rowSums(X) + 6))

  } else if(nam %in% c("gaussianinverse")){
    sds <- p:1
    sds <- sqrt(sds^2 / sum(sds^2))
    X <- matrix(nrow = n, ncol = p)
    set.seed(77311413)
    for(i in 1:p)
      X[, i] <- rnorm(n, 0, sds[i])
    y <- rnorm(n, family$linkinv(rowSums(X)))

  }

  list(X = X, y = y)
}

test_that("works with different families", {
  n <- 1000L
  p <- 5L
  set.seed(77311413)
  for(fa in list(
    binomial("logit"), binomial("probit"), binomial("cauchit"),
    binomial("cloglog"), gaussian("identity"), gaussian("inverse"),
    gaussian("log"))){
    tmp <- sim_func(fa, n, p)
    X <- tmp$X
    y <- tmp$y

    #####
    # no weights, no offset
    lab <- paste0(fa$family, "_", fa$link)
    suppressWarnings({
      f2 <- glm(y ~ X, family = fa)
      f1 <- parglm(y ~ X, family = fa, control = parglm.control(nthreads = 2))
    })

    expect_equal(f1[to_check], f2[to_check], label = lab)
    # these may differ as `glm.fit` uses the weights from the iteration prior
    # to convergence
    expect_equal(f1$weights, f2$weights, tolerance = sqrt(1e-7), label = lab)

    s2 <- summary(f2)
    s1 <- summary(f1)

    excl <- c("call", "coefficients", "cov.unscaled", "cov.scaled",
              "dispersion")
    expect_equal(s1[!names(s1) %in% excl], s2[!names(s2) %in% excl],
                 label = lab, tolerance = 1e-7)
    na <- rownames(s1$coefficients)
    expect_equal(s1$coefficients[na, 1:2], s2$coefficients[na, 1:2],
                 label = lab, tolerance = 1e-7)

    # may also differ as the weights are not computed at the final estimates
    expect_equal(s1$dispersion, s2$dispersion, label = lab,
                 tolerance = sqrt(1e-7))

    #####
    # no weights, offset
    lab <- paste0(fa$family, "_", fa$link, " w/ offset")
    suppressWarnings({
      f2 <- glm(y ~ X[, -1], family = fa, offset = X[, 1])
      f1 <- parglm(y ~ X[, -1], family = fa, offset = X[, 1],
                   control = parglm.control(nthreads = 2))
    })

    expect_equal(f1[to_check], f2[to_check], label = lab)
    # these may differ as `glm.fit` uses the weights from the iteration prior
    # to convergence
    expect_equal(f1$weights, f2$weights, tolerance = sqrt(1e-7), label = lab)

    s2 <- summary(f2)
    s1 <- summary(f1)

    expect_equal(s1[!names(s1) %in% excl], s2[!names(s2) %in% excl],
                 label = lab)
    na <- rownames(s1$coefficients)
    expect_equal(s1$coefficients[na, 1:2], s2$coefficients[na, 1:2],
                 label = lab, tolerance = 1e-7)

    # may also differ as the weights are not computed at the final estimates
    expect_equal(s1$dispersion, s2$dispersion, label = lab,
                 tolerance = sqrt(1e-7))

    #####
    # weights, no offset
    lab <- paste0(fa$family, "_", fa$link, " w/ weigths")
    w <- runif(n)
    w <- n * w / sum(w)
    suppressWarnings({
      f2 <- glm(y ~ X, family = fa, weights = w)
      f1 <- parglm(y ~ X, family = fa, weights = w,
                   control = parglm.control(nthreads = 2))
    })

    expect_equal(f1[to_check], f2[to_check], label = lab)
    # these may differ as `glm.fit` uses the weights from the iteration prior
    # to convergence
    expect_equal(f1$weights, f2$weights, tolerance = sqrt(1e-7), label = lab)

    s2 <- summary(f2)
    s1 <- summary(f1)

    expect_equal(s1[!names(s1) %in% excl], s2[!names(s2) %in% excl],
                 label = lab)
    na <- rownames(s1$coefficients)
    expect_equal(s1$coefficients[na, 1:2], s2$coefficients[na, 1:2],
                 label = lab, tolerance = 1e-7)

    # may also differ as the weights are not computed at the final estimates
    expect_equal(s1$dispersion, s2$dispersion, label = lab,
                 tolerance = sqrt(1e-7))
  }
})




n <- 1000L
p <- 20L
li <- binomial("logit")
sds <- p:1
sds <- sqrt(sds^2 / sum(sds^2))
X <- matrix(nrow = n, ncol = p)
for(i in 1:p)
  X[, i] <- rnorm(n, 1/p, sds[i])
y <- 1/(1 + exp(-(1 - rowSums(X)))) > runif(n)

microbenchmark::microbenchmark(
  f1 = f1 <- glm(y ~ X, binomial()),
  f2.1 = f2 <- parglm(y ~ X, binomial(), control = parglm.control(nthreads = 1)),
  f2.2 = f2 <- parglm(y ~ X, binomial(), control = parglm.control(nthreads = 2)),
  f2.4 = f2 <- parglm(y ~ X, binomial(), control = parglm.control(nthreads = 4)),
  times = 10)

f2.4 = f2 <- parglm(y ~ X, binomial(), control = parglm.control(nthreads = 4))

qr.R(f2.4$qr)
