## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3
* Ubuntu 18.04 LTS with gcc 8.3.0 with `--use-valgrind`
  R version 3.6.3
* Ubuntu 16.04 LTS (on travis-ci)
  R version 4.0.0
* Ubuntu 18.04 LTS with clang 6.0.0 with ASAN and 
  UBSAN checks
  R devel (2020-07-02 r78770)
* win-builder (devel and release)
* `rhub::check_on_solaris()`
* `rhub::check_for_cran()`
* Fedora 30 with openBLAS like on CRAN's issue kinds page.
  
## R CMD check results
There is a note about the size of the package on some platforms.

I have tried to reproduce the error with openBLAS and Solaris but failed 
to do so. I made a post on R-pkg-devel 
(https://stat.ethz.ch/pipermail/r-package-devel/2020q1/005175.html)
but I did not get a reply.

## Resubmission
This is a resubmission. In this version I have:

 - made the tests more stable numerically.
