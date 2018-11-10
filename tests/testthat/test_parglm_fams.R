context("testing results with varrious link functions agianst `glm.fit`")

to_check <- c(
  "coefficients", "residuals", "fitted.values", "rank",
  "family", "linear.predictors", "deviance", "aic", "null.deviance",
  "iter", "prior.weights", "df.residual", "df.null",
  "converged", "boundary", "formula", "terms", "data",
  "offset", "contrasts", "xlevels")

n <- 1000L
p <- 5L
li <- binomial("logit")
sds <- p:1
sds <- sqrt(sds^2 / sum(sds^2))
X <- matrix(nrow = n, ncol = p)
for(i in 1:p)
  X[, i] <- rnorm(n, 1/p, sds[i])
y <- 1/(1 + exp(-(1 - rowSums(X)))) > runif(n)

f1 <- parglm(y ~ X, family = binomial(),
             control = parglm.control(nthreads = 2, block_size = 250))
f2 <- glm(y ~ X, family = binomial())

expect_equal(f1[to_check], f2[to_check], tolerance = 1e-7)
# differs as one is caculated after final it
expect_equal(f1$weights, f2$weights, tolerance = 1e-4)

s2 <- summary(f2)
s1 <- summary(f1)

excl <- c("call", "coefficients", "cov.unscaled")
expect_equal(s1[!names(s1) %in% excl], s2[!names(s2) %in% excl])
expect_equal(s1$coefficients[, 1:2], s2$coefficients[, 1:2], tol = 1e-5)

