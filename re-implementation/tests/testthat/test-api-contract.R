# API contract for MAGICAL_estimation.
# These tests describe the shape and value invariants any backend must satisfy.
# They exercise the current (R) backend now; the C++ backend must pass them
# unmodified once wired up.

context("MAGICAL_estimation API contract")

test_that("MAGICAL_estimation returns the documented list shape", {
  fx <- make_synthetic_fixture(seed = 1L)
  n_iter <- 20L

  set.seed(101)
  out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                            fx$Initial_model, iteration_num = n_iter,
                            backend = "r")

  expect_named(out, c("TF_Peak_Binding_prob",
                      "Peak_Gene_Looping_prob",
                      "Noise_parameters"),
               ignore.order = TRUE)

  P <- nrow(fx$Candidate_circuits$Peaks)
  M <- nrow(fx$Candidate_circuits$TFs)
  G <- nrow(fx$Candidate_circuits$Genes)

  expect_equal(dim(out$TF_Peak_Binding_prob),  c(P, M))
  expect_equal(dim(out$Peak_Gene_Looping_prob), c(P, G))
  expect_equal(dim(out$Noise_parameters),      c(n_iter, 2L))
})

test_that("posterior probabilities lie in [0, 1] and are finite", {
  fx <- make_synthetic_fixture(seed = 2L)

  set.seed(202)
  out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                            fx$Initial_model, iteration_num = 25L,
                            backend = "r")

  for (mat_name in c("TF_Peak_Binding_prob", "Peak_Gene_Looping_prob")) {
    m <- as.matrix(out[[mat_name]])
    expect_true(all(is.finite(m)),        info = mat_name)
    expect_true(all(m >= 0 & m <= 1),     info = mat_name)
  }
})

test_that("Noise_parameters are strictly positive variances", {
  fx <- make_synthetic_fixture(seed = 3L)

  set.seed(303)
  out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                            fx$Initial_model, iteration_num = 30L,
                            backend = "r")

  expect_true(all(is.finite(out$Noise_parameters)))
  expect_true(all(out$Noise_parameters > 0))
})

test_that("posterior mass concentrates on candidate (non-zero prior) entries", {
  # Peaks/genes with zero prior support in the candidate matrices should never
  # gain posterior mass, since the sampler only updates entries with prior > 0.
  fx <- make_synthetic_fixture(seed = 4L)

  set.seed(404)
  out <- MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                            fx$Initial_model, iteration_num = 15L,
                            backend = "r")

  binding_prior <- as.matrix(fx$Candidate_circuits$TF_Peak_Binding)
  looping_prior <- as.matrix(fx$Candidate_circuits$Peak_Gene_looping)

  expect_true(all(out$TF_Peak_Binding_prob[binding_prior == 0] == 0))
  expect_true(all(out$Peak_Gene_Looping_prob[looping_prior == 0] == 0))
})
