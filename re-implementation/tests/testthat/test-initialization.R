# Tests for MAGICAL_initialization.

context("MAGICAL_initialization")

test_that("returns a list with the documented 11 named fields", {
  fx <- make_synthetic_fixture(seed = 11L)
  init <- fx$Initial_model

  expected <- c("T_prior", "T_mean", "T_var",
                "B_prior", "B_mean", "B_var", "B_prob",
                "L_prior", "L_mean", "L_var", "L_prob")
  expect_named(init, expected, ignore.order = TRUE)
})

test_that("T_prior equals input TF_log2Count", {
  fx <- make_synthetic_fixture(seed = 12L)
  expect_equal(fx$Initial_model$T_prior, fx$Candidate_circuits$TF_log2Count)
})

test_that("T_mean equals input TF_log2Count", {
  fx <- make_synthetic_fixture(seed = 13L)
  expect_equal(fx$Initial_model$T_mean, fx$Candidate_circuits$TF_log2Count)
})

test_that("T_var has shape (M, S) and finite non-negative values", {
  fx <- make_synthetic_fixture(seed = 14L)
  M <- nrow(fx$Candidate_circuits$TFs)
  S <- length(fx$loaded_data$Common_samples)
  expect_equal(dim(fx$Initial_model$T_var), c(M, S))
  expect_true(all(is.finite(fx$Initial_model$T_var)))
  expect_true(all(fx$Initial_model$T_var >= 0))
})

test_that("B_prior has shape (P, M) and zero where the candidate prior is zero", {
  fx <- make_synthetic_fixture(seed = 15L)
  P <- nrow(fx$Candidate_circuits$Peaks)
  M <- nrow(fx$Candidate_circuits$TFs)
  expect_equal(dim(fx$Initial_model$B_prior), c(P, M))

  cand <- as.matrix(fx$Candidate_circuits$TF_Peak_Binding)
  expect_true(all(fx$Initial_model$B_prior[cand == 0] == 0))
})

test_that("B_prob has values in [0, 1] and is zero outside the candidate mask", {
  fx <- make_synthetic_fixture(seed = 16L)
  bp <- fx$Initial_model$B_prob
  expect_true(all(is.finite(bp)))
  expect_true(all(bp >= 0 & bp <= 1))

  cand <- as.matrix(fx$Candidate_circuits$TF_Peak_Binding)
  expect_true(all(bp[cand == 0] == 0))
})

test_that("B_var has shape (1, M) and finite positive values", {
  fx <- make_synthetic_fixture(seed = 17L)
  M <- nrow(fx$Candidate_circuits$TFs)
  expect_equal(dim(fx$Initial_model$B_var), c(1L, M))
  expect_true(all(is.finite(fx$Initial_model$B_var)))
  expect_true(all(fx$Initial_model$B_var > 0))
})

test_that("L_prior has shape (P, G) and zero where the candidate prior is zero", {
  fx <- make_synthetic_fixture(seed = 18L)
  P <- nrow(fx$Candidate_circuits$Peaks)
  G <- nrow(fx$Candidate_circuits$Genes)
  expect_equal(dim(fx$Initial_model$L_prior), c(P, G))

  cand <- as.matrix(fx$Candidate_circuits$Peak_Gene_looping)
  expect_true(all(fx$Initial_model$L_prior[cand == 0] == 0))
})

test_that("L_prob has values in [0, 1] and is zero outside the candidate mask", {
  fx <- make_synthetic_fixture(seed = 19L)
  lp <- fx$Initial_model$L_prob
  expect_true(all(is.finite(lp)))
  expect_true(all(lp >= 0 & lp <= 1))

  cand <- as.matrix(fx$Candidate_circuits$Peak_Gene_looping)
  expect_true(all(lp[cand == 0] == 0))
})

test_that("L_var is a finite non-negative scalar", {
  fx <- make_synthetic_fixture(seed = 20L)
  lv <- fx$Initial_model$L_var
  expect_true(length(lv) == 1L)
  expect_true(is.finite(lv))
  expect_true(lv >= 0)
})

test_that("initialization is deterministic under the same seed", {
  # MAGICAL_initialization has no RNG, so it should be a pure function of inputs.
  fx1 <- make_synthetic_fixture(seed = 21L)
  fx2 <- make_synthetic_fixture(seed = 21L)
  expect_equal(fx1$Initial_model, fx2$Initial_model)
})
