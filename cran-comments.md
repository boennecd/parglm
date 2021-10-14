## Test environments
* Ubuntu 20.04 LTS with gcc 10.1.0
  R version 4.1.1
* Ubuntu 20.04 LTS with gcc 10.1.0
  R version 4.1.1 with valgrind
* Ubuntu 20.04 LTS with gcc 10.1.0
  R devel 2021-10-09 r81024 with ASAN and UBSAN
* Github actions on windows-latest (release), macOS-latest (release), 
  ubuntu-20.04 (release), and ubuntu-20.04 (devel)
* win-builder (devel, oldrelease, and release)
* `rhub::check_for_cran()`
* `rhub::check(platform = c("fedora-clang-devel", "macos-highsierra-release-cran"))`
  
## R CMD check results
There were no WARNINGs or ERRORs.

There is a NOTE about the package size in some cases.
