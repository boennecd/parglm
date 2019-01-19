context("Miscellaneous tests")

test_that("'parglm' works when package is not attached",{
  # Issue: https://github.com/boennecd/parglm/issues/2#issue-397286510
  # See https://github.com/r-lib/devtools/issues/1797#issuecomment-423288947

  expect_silent(
    local({
      detach("package:parglm", unload = TRUE, force = TRUE)
      parglm::parglm(mpg ~ gear , data = datasets::mtcars)
      library(parglm)
    },
    envir= new.env(parent = environment(glm))))
})
