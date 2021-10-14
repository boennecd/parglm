# parglm 0.1.7
* avoid some virtual function calls and remove a few macros. 
* fix an issue due to new STRICT_R_HEADERS variable in Rcpp.

# parglm 0.1.6
* avoid some memory allocations.
* fix a test issue with ATLAS.

# parglm 0.1.4
* `stop`s when there are more variables than observations. Previously, this 
  caused a crash.
* handle Fortran string length argument.

# parglm 0.1.3
* Fix bug found with Valgrind.

# parglm 0.1.2
* Minor changes in implementation.
* Fix bugs in patched R and oldrel R.

# parglm 0.1.1
* A `FAST` method is added which computes the Fisher information and then solves 
  the normal equation as in `speedglm`. 
* One change which decreased the computation time.
* Minor bug fixes.
