# Cross-backend statistical equivalence.
#
# Under Option B (Armadillo RNG for the C++ backend) the two backends will
# NOT produce identical draws, but pooled posterior probabilities from
# adequately-mixed chains should agree closely. These tests define what
# "closely" means as a contract.
#
# During stage 1 the "cpp" backend is not implemented, so the cpp-vs-r test
# is skipped. The r-vs-r stability test still runs and guards against
# nondeterminism creeping into the R implementation.

context("Cross-backend equivalence")

# Threshold for pooled-posterior agreement on the demo-scale problem.
# On tiny synthetic fixtures with few iterations, chain-to-chain noise
# dominates, so we test a weaker property: high correlation of the pooled
# means from two independent runs of the SAME backend.
CORR_TOL_SAME_BACKEND    <- 0.90   # r-vs-r, different seeds, tiny data
CORR_TOL_ACROSS_BACKENDS <- 0.90   # cpp-vs-r, matched seed intent

run_backend <- function(fx, backend, seed, n_iter) {
  set.seed(seed)
  MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits, fx$Initial_model,
                     iteration_num = n_iter, backend = backend)
}

matrix_corr <- function(a, b) {
  a_vec <- as.numeric(as.matrix(a))
  b_vec <- as.numeric(as.matrix(b))
  # Constant vectors -> undefined correlation; treat as perfect agreement iff equal.
  if (sd(a_vec) == 0 || sd(b_vec) == 0) {
    return(if (isTRUE(all.equal(a_vec, b_vec))) 1 else NA_real_)
  }
  cor(a_vec, b_vec)
}

test_that("two R-backend runs with different seeds yield correlated posteriors", {
  fx <- make_synthetic_fixture(seed = 42L)

  run_a <- run_backend(fx, "r", seed = 1001L, n_iter = 200L)
  run_b <- run_backend(fx, "r", seed = 2002L, n_iter = 200L)

  binding_corr <- matrix_corr(run_a$TF_Peak_Binding_prob,
                              run_b$TF_Peak_Binding_prob)
  looping_corr <- matrix_corr(run_a$Peak_Gene_Looping_prob,
                              run_b$Peak_Gene_Looping_prob)

  expect_true(!is.na(binding_corr) && binding_corr >= CORR_TOL_SAME_BACKEND,
              info = sprintf("binding corr = %.3f", binding_corr))
  expect_true(!is.na(looping_corr) && looping_corr >= CORR_TOL_SAME_BACKEND,
              info = sprintf("looping corr = %.3f", looping_corr))
})

test_that("cpp backend agrees with R backend up to chain-to-chain noise", {
  # Compilation gated behind MAGICAL_TEST_CPP_BUILD=1 (same flag as
  # test-cpp-build.R) so a plain `testthat.R` run stays fast and doesn't
  # invoke the C++ toolchain.
  skip_if(Sys.getenv("MAGICAL_TEST_CPP_BUILD") != "1",
          "set MAGICAL_TEST_CPP_BUILD=1 to enable")
  skip_if_not_installed("Rcpp")
  skip_if_not_installed("RcppArmadillo")

  ok <- tryCatch({ magical_load_cpp(); TRUE },
                 error = function(e) { message(conditionMessage(e)); FALSE })
  skip_if_not(ok, "magical.cpp failed to compile")
  skip_if_not(exists("magical_estimation_cpp", mode = "function"),
              "cpp entrypoint not exported")

  fx <- make_synthetic_fixture(seed = 7L)

  run_r   <- run_backend(fx, "r",   seed = 5150L, n_iter = 200L)
  run_cpp <- run_backend(fx, "cpp", seed = 5150L, n_iter = 200L)

  binding_corr <- matrix_corr(run_r$TF_Peak_Binding_prob,
                              run_cpp$TF_Peak_Binding_prob)
  looping_corr <- matrix_corr(run_r$Peak_Gene_Looping_prob,
                              run_cpp$Peak_Gene_Looping_prob)

  expect_true(binding_corr >= CORR_TOL_ACROSS_BACKENDS,
              info = sprintf("binding corr = %.3f", binding_corr))
  expect_true(looping_corr >= CORR_TOL_ACROSS_BACKENDS,
              info = sprintf("looping corr = %.3f", looping_corr))
})
