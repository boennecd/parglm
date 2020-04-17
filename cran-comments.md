## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.1
* Ubuntu 16.04 LTS (on travis-ci)
  R version 3.6.1
* Ubuntu 18.04 LTS with gcc 8.3.0 with --enable-lto
  R devel (2019-11-06 r77376)
* Ubuntu 18.04 LTS with clang 6.0.0 with ASAN and 
  UBSAN checks
  R devel (2019-11-06 r77376)
* win-builder (devel and release)
* `rhub::check_on_solaris()` when it was still up 
* `devtools::check_win_release()`
* Fedora 30 with openBLAS like on CRAN
  
## R CMD check results
There is a note about the size of the package on some platforms.

I have tried to reproduce the error with openBLAS and Solaris but failed 
to do so. I made a post on R-pkg-devel 
(https://stat.ethz.ch/pipermail/r-package-devel/2020q1/005175.html)
but I did not get a reply.
