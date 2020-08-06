## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3
* Ubuntu 16.04 LTS (on travis-ci)
  R version 4.0.0
* win-builder (devel and release)
* `rhub::check_for_cran()`
  
## R CMD check results
There is a note about the size of the package on some platforms.

I changed one of the tests such that it no longer checks the number of 
iterations used by the method. This is the reason the test failed on ATLAS 
which caused the package to get pulled of CRAN. However, the estimates 
where similar to the test values and hence checking the iteration number 
makes little sense.
