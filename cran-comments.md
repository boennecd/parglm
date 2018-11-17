## Test environments
* local Windows 10 machine with R 3.5.0
* win-builder (devel and release)
* Ubuntu 14.04 (on travis-ci), R 3.5.1
* Ubuntu 17.10 with clang 6.0.0, devel, and ASAN/UBSAN settings
* The following rhub platforms:
  debian-gcc-devel
  fedora-clang-devel
  fedora-gcc-devel
  debian-gcc-patched
  debian-gcc-release
  solaris-x86-patched
  linux-x86_64-rocker-gcc-san

## R CMD check results
There were no ERRORs, or WARNINGs. There is a NOTE on `fedora-gcc-devel` and
`debian-gcc-release` on rhub about the size of the package.
