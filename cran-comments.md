## Test environments
* Ubuntu 18.04 LTS
  R version 3.5.3
* Ubuntu 14.04.5 LTS (on travis-ci with codename: trusty)
  R version 3.5.2
* win-builder (devel and release)
* Local Ubuntu 18.04 with R 3.5.2 and with clang 6.0.0 with ASAN and 
  UBSAN checks
* The following rhub platforms:
  fedora-clang-devel
  fedora-gcc-devel
  debian-gcc-patched
  debian-gcc-devel
  debian-gcc-release
  linux-x86_64-rocker-gcc-san
  solaris-x86-patched

## R CMD check results
There were no ERRORs, or WARNINGs. There is a NOTE on `fedora-gcc-devel`
on rhub, my local machine, and `debian-gcc-release` on rhub about the size 
of the package.
