# Additional MAGICAL_estimation edge-case and property tests.

context("MAGICAL_estimation additional properties")

test_that("small iteration_num runs without error and returns valid output", {
  # Note: the original code has a latent bug for iteration_num < 6
  # (round(iteration_num/10) becomes 0, then i %% 0 is NaN). We start at 6.
  fx <- make_synthetic_fixture(seed = 50L)
  set.seed(1)
  out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                            fx$Initial_model, iteration_num = 6L,
                            backend = "r")
  expect_equal(dim(out$Noise_parameters), c(6L, 2L))
  expect_true(all(is.finite(as.matrix(out$TF_Peak_Binding_prob))))
  expect_true(all(is.finite(as.matrix(out$Peak_Gene_Looping_prob))))
})

test_that("MAGICAL_estimation is deterministic under the same seed", {
  fx <- make_synthetic_fixture(seed = 51L)
  set.seed(999)
  a <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                          fx$Initial_model, iteration_num = 15L,
                          backend = "r")
  set.seed(999)
  b <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                          fx$Initial_model, iteration_num = 15L,
                          backend = "r")
  expect_equal(a$TF_Peak_Binding_prob,   b$TF_Peak_Binding_prob)
  expect_equal(a$Peak_Gene_Looping_prob, b$Peak_Gene_Looping_prob)
  expect_equal(a$Noise_parameters,       b$Noise_parameters)
})

test_that("Noise_parameters row count tracks iteration_num", {
  # iteration_num >= 6 to avoid the round(n/10) = 0 code path.
  fx <- make_synthetic_fixture(seed = 52L)
  for (n in c(6L, 15L, 25L)) {
    set.seed(100L + n)
    out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                              fx$Initial_model, iteration_num = n,
                              backend = "r")
    expect_equal(nrow(out$Noise_parameters), n,
                 info = sprintf("iteration_num = %d", n))
  }
})

test_that("different seeds produce different noise trajectories", {
  # Sanity check: proves the sampler actually uses R's RNG state.
  fx <- make_synthetic_fixture(seed = 53L)
  set.seed(1)
  a <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                          fx$Initial_model, iteration_num = 10L,
                          backend = "r")
  set.seed(2)
  b <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                          fx$Initial_model, iteration_num = 10L,
                          backend = "r")
  expect_false(isTRUE(all.equal(a$Noise_parameters, b$Noise_parameters)))
})

test_that("backend = 'cpp' returns the same shapes as backend = 'r'", {
  # Stage 2: cpp backend now works. Compare shapes + name preservation.
  # Numerical equivalence is covered separately in test-equivalence.R.
  skip_if(Sys.getenv("MAGICAL_TEST_CPP_BUILD") != "1",
          "set MAGICAL_TEST_CPP_BUILD=1 to enable")
  skip_if_not_installed("Rcpp")
  skip_if_not_installed("RcppArmadillo")
  ok <- tryCatch({ magical_load_cpp(); TRUE }, error = function(e) FALSE)
  skip_if_not(ok, "magical.cpp failed to compile")

  fx <- make_synthetic_fixture(seed = 54L)
  set.seed(1)
  r_out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                              fx$Initial_model, iteration_num = 10L,
                              backend = "r")
  cpp_out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                                fx$Initial_model, iteration_num = 10L,
                                backend = "cpp", seed = 1L)

  expect_equal(dim(cpp_out$TF_Peak_Binding_prob),
               dim(r_out$TF_Peak_Binding_prob))
  expect_equal(dim(cpp_out$Peak_Gene_Looping_prob),
               dim(r_out$Peak_Gene_Looping_prob))
  expect_equal(dim(cpp_out$Noise_parameters),
               dim(r_out$Noise_parameters))
  expect_equal(dimnames(cpp_out$TF_Peak_Binding_prob),
               dimnames(r_out$TF_Peak_Binding_prob))
  expect_equal(dimnames(cpp_out$Peak_Gene_Looping_prob),
               dimnames(r_out$Peak_Gene_Looping_prob))
})

test_that("posterior probabilities are bounded by (state_frq_max)/(iter+1) = 1", {
  # The average of binary states over (iter+1) steps must lie in [0, 1].
  # Complementary to test-api-contract; explicit iteration-count arithmetic.
  fx <- make_synthetic_fixture(seed = 55L)
  n_iter <- 8L
  set.seed(1)
  out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                            fx$Initial_model, iteration_num = n_iter,
                            backend = "r")

  binding <- as.matrix(out$TF_Peak_Binding_prob)
  looping <- as.matrix(out$Peak_Gene_Looping_prob)
  # every value is a fraction with denominator (iter+1), so multiplying
  # by (iter+1) should give an integer count in [0, iter+1]
  binding_counts <- binding * (n_iter + 1)
  looping_counts <- looping * (n_iter + 1)
  expect_true(all(abs(binding_counts - round(binding_counts)) < 1e-9))
  expect_true(all(abs(looping_counts - round(looping_counts)) < 1e-9))
  expect_true(all(binding_counts >= 0 & binding_counts <= n_iter + 1))
  expect_true(all(looping_counts >= 0 & looping_counts <= n_iter + 1))
})
