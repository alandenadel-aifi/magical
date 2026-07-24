# Verifies the Rcpp + RcppArmadillo toolchain compiles src/magical.cpp.
# Skipped unless MAGICAL_TEST_CPP_BUILD=1 (opt-in because it invokes the
# system C++ compiler and can be slow / require BuildTools).

context("C++ toolchain build probe")

test_that("magical.cpp compiles and exports magical_cpp_probe()", {
  skip_if(Sys.getenv("MAGICAL_TEST_CPP_BUILD") != "1",
          "set MAGICAL_TEST_CPP_BUILD=1 to enable")

  skip_if_not_installed("Rcpp")
  skip_if_not_installed("RcppArmadillo")

  magical_load_cpp()

  expect_true(exists("magical_cpp_probe", mode = "function"))
  probe <- magical_cpp_probe()
  expect_type(probe, "character")
  expect_match(probe, "magical\\.cpp loaded")
})
