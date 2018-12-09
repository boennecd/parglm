[![Build Status on Travis](https://travis-ci.org/boennecd/parglm.svg?branch=master)](https://travis-ci.org/boennecd/parglm)
[![](https://www.r-pkg.org/badges/version/parglm)](https://CRAN.R-project.org/package=parglm)
[![CRAN RStudio mirror downloads](http://cranlogs.r-pkg.org/badges/parglm)](https://CRAN.R-project.org/package=parglm)

parglm
======

The `parglm` package provides a parallel estimation method  for generalized 
linear models without compiling with a multithreaded LAPACK or BLAS. You can install
it from Github by calling:

```r
devtools::install_github("boennecd/parglm")
```

or from CRAN by calling:

```r
install.packages("parglm")
```

See the [vignettes/parglm.html](https://htmlpreview.github.io/?https://github.com/boennecd/parglm/blob/master/vignettes/parglm.html) for an example of run times and 
further details.
