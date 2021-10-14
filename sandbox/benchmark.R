#####
options(digits = 3L)

library(parglm)
n <- 20000L
p <- 100L
li <- binomial("logit")
sds <- p:1
sds <- sqrt(sds^2 / sum(sds^2))
set.seed(13103921)
X <- matrix(nrow = n, ncol = p)
for(i in 1:p)
  X[, i] <- rnorm(n, 1/p, sds[i])
y <- 1/(1 + exp(-(1 - rowSums(X)))) > runif(n)

fit_func <- function(ctrl)
  parglm(y ~ X, binomial(), control = ctrl)

library(microbenchmark)
mic <- microbenchmark(
  glm  = glm_fit <- glm(y ~ X, binomial()),

  `QR 1` = fit_func(parglm.control(nthreads = 1)),
  `QR 2` = fit_func(parglm.control(nthreads = 2)),
  `QR 4` = fit_func(parglm.control(nthreads = 4)),

  `fast 1` = f4 <- fit_func(parglm.control(nthreads = 1, method = "FAST")),
  `fast 2` = f4 <- fit_func(parglm.control(nthreads = 2, method = "FAST")),
  `fast 4` = f4 <- fit_func(parglm.control(nthreads = 4, method = "FAST")),
  times = 5, setup = gc())
summary(mic)[, c("expr", "lq", "median", "uq")]

b_size <- n / 4L / 5L
v <- microbenchmark(
  glm  = glm_fit <- glm(y ~ X, binomial()),
  `QR 1` = fit_func(parglm.control(nthreads = 1, block_size = b_size)),
  `QR 2` = fit_func(parglm.control(nthreads = 2, block_size = b_size)),
  `QR 4` = fit_func(parglm.control(nthreads = 4, block_size = b_size)),

  `fast 1` = fit_func(parglm.control(
    nthreads = 1, method = "FAST", block_size = b_size)),
  `fast 2` = fit_func(parglm.control(
    nthreads = 2, method = "FAST", block_size = b_size)),
  `fast 4` = fit_func(parglm.control(
    nthreads = 4, method = "FAST", block_size = b_size)),
  times = 5, setup = gc())
summary(mic)[, c("expr", "lq", "median", "uq")]
