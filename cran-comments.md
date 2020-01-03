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
* `rhub::check_on_solaris()`
  
## R CMD check results
The LTO issue have been solved. There is a note about the size of the 
package on some platforms.
