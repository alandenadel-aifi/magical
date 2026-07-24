# Unit tests for the six MCMC samplers, called directly (bypassing the
# public MAGICAL_estimation driver).
#
# All tests hit the ORIGINAL R samplers via the private env, so they pass
# right now. The C++ ports in stage 2 will need to satisfy the same
# structural contracts (with statistical rather than bit-identical values).

context("Sampler: TF_activity_T_sampling")

test_that("TFA output has correct component shapes", {
  fx <- make_synthetic_fixture(seed = 30L)
  args <- make_sampler_inputs(fx, seed = 30L)

  set.seed(1)
  out <- orig("TF_activity_T_sampling")(
    args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
    args$R, args$R_sample, args$RNA_Cell_Sample_vector,
    args$TFA, args$T_prior_mean, args$T_prior_var,
    args$B, args$B_state, args$sigma_A_noise,
    args$P, args$G, args$M, args$S
  )

  expect_named(out, c("T_A", "T_R", "T_sample"), ignore.order = TRUE)
  expect_equal(dim(out$T_A),      dim(args$TFA$T_A))
  expect_equal(dim(out$T_R),      dim(args$TFA$T_R))
  expect_equal(dim(out$T_sample), dim(args$TFA$T_sample))
})

test_that("TFA output is finite everywhere", {
  fx <- make_synthetic_fixture(seed = 31L)
  args <- make_sampler_inputs(fx, seed = 31L)

  set.seed(2)
  out <- orig("TF_activity_T_sampling")(
    args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
    args$R, args$R_sample, args$RNA_Cell_Sample_vector,
    args$TFA, args$T_prior_mean, args$T_prior_var,
    args$B, args$B_state, args$sigma_A_noise,
    args$P, args$G, args$M, args$S
  )
  expect_true(all(is.finite(out$T_A)))
  expect_true(all(is.finite(out$T_R)))
  expect_true(all(is.finite(out$T_sample)))
})

test_that("TF_activity_T_sampling is deterministic under the same seed", {
  fx <- make_synthetic_fixture(seed = 32L)
  args <- make_sampler_inputs(fx, seed = 32L)
  f <- orig("TF_activity_T_sampling")

  set.seed(3)
  a <- f(args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
         args$R, args$R_sample, args$RNA_Cell_Sample_vector,
         args$TFA, args$T_prior_mean, args$T_prior_var,
         args$B, args$B_state, args$sigma_A_noise,
         args$P, args$G, args$M, args$S)
  set.seed(3)
  b <- f(args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
         args$R, args$R_sample, args$RNA_Cell_Sample_vector,
         args$TFA, args$T_prior_mean, args$T_prior_var,
         args$B, args$B_state, args$sigma_A_noise,
         args$P, args$G, args$M, args$S)
  expect_equal(a, b)
})


context("Sampler: TF_peak_binding_B_sampling")

test_that("returns a (P, M) matrix with the same shape as B", {
  fx <- make_synthetic_fixture(seed = 33L)
  args <- make_sampler_inputs(fx, seed = 33L)
  set.seed(4)
  B_new <- orig("TF_peak_binding_B_sampling")(
    args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
    args$TFA, args$B, args$B_state,
    args$B_prior_mean, args$B_prior_var, args$sigma_A_noise,
    args$P, args$G, args$M, args$S
  )
  expect_equal(dim(B_new), dim(args$B))
})

test_that("B is zero wherever B_state is zero", {
  fx <- make_synthetic_fixture(seed = 34L)
  args <- make_sampler_inputs(fx, seed = 34L)
  set.seed(5)
  B_new <- orig("TF_peak_binding_B_sampling")(
    args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
    args$TFA, args$B, args$B_state,
    args$B_prior_mean, args$B_prior_var, args$sigma_A_noise,
    args$P, args$G, args$M, args$S
  )
  expect_true(all(B_new[args$B_state == 0] == 0))
  expect_true(all(is.finite(B_new)))
})

test_that("TF_peak_binding_B_sampling is deterministic under the same seed", {
  fx <- make_synthetic_fixture(seed = 35L)
  args <- make_sampler_inputs(fx, seed = 35L)
  f <- orig("TF_peak_binding_B_sampling")
  set.seed(6)
  a <- f(args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
         args$TFA, args$B, args$B_state,
         args$B_prior_mean, args$B_prior_var, args$sigma_A_noise,
         args$P, args$G, args$M, args$S)
  set.seed(6)
  b <- f(args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
         args$TFA, args$B, args$B_state,
         args$B_prior_mean, args$B_prior_var, args$sigma_A_noise,
         args$P, args$G, args$M, args$S)
  expect_equal(a, b)
})


context("Sampler: TF_peak_binary_binding_B_state_sampling")

test_that("returns a list with B and B_state of shape (P, M)", {
  fx <- make_synthetic_fixture(seed = 36L)
  args <- make_sampler_inputs(fx, seed = 36L)
  set.seed(7)
  out <- orig("TF_peak_binary_binding_B_state_sampling")(
    args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
    args$TFA, args$B, args$B_state,
    args$B_prior_mean, args$B_prior_var, args$B_prior_prob,
    args$sigma_A_noise, args$P, args$G, args$M, args$S
  )
  expect_named(out, c("B", "B_state"), ignore.order = TRUE)
  expect_equal(dim(out$B),       dim(args$B))
  expect_equal(dim(out$B_state), dim(args$B_state))
})

test_that("B_state values are strictly binary in {0, 1}", {
  fx <- make_synthetic_fixture(seed = 37L)
  args <- make_sampler_inputs(fx, seed = 37L)
  set.seed(8)
  out <- orig("TF_peak_binary_binding_B_state_sampling")(
    args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
    args$TFA, args$B, args$B_state,
    args$B_prior_mean, args$B_prior_var, args$B_prior_prob,
    args$sigma_A_noise, args$P, args$G, args$M, args$S
  )
  expect_true(all(out$B_state %in% c(0, 1)))
})

test_that("B_state stays zero where the prior probability is zero", {
  fx <- make_synthetic_fixture(seed = 38L)
  args <- make_sampler_inputs(fx, seed = 38L)
  set.seed(9)
  out <- orig("TF_peak_binary_binding_B_state_sampling")(
    args$A, args$A_sample, args$ATAC_Cell_Sample_vector,
    args$TFA, args$B, args$B_state,
    args$B_prior_mean, args$B_prior_var, args$B_prior_prob,
    args$sigma_A_noise, args$P, args$G, args$M, args$S
  )
  # entries with B_prior_prob == 0 and B_state == 0 stay 0 (sampler skips them)
  mask <- args$B_prior_prob == 0 & args$B_state == 0
  expect_true(all(out$B_state[mask] == 0))
  expect_true(all(out$B[mask]       == 0))
})


context("Sampler: Peak_gene_looping_L_samping")

test_that("returns a (P, G) matrix with the same shape as L", {
  fx <- make_synthetic_fixture(seed = 39L)
  args <- make_sampler_inputs(fx, seed = 39L)
  set.seed(10)
  L_new <- orig("Peak_gene_looping_L_samping")(
    args$R, args$R_sample, args$RNA_Cell_Sample_vector,
    args$TFA, args$B, args$L, args$L_state,
    args$L_prior_mean, args$L_prior_var, args$sigma_R_noise,
    args$P, args$G, args$M, args$S
  )
  expect_equal(dim(L_new), dim(args$L))
})

test_that("L is zero wherever L_state is zero", {
  fx <- make_synthetic_fixture(seed = 40L)
  args <- make_sampler_inputs(fx, seed = 40L)
  set.seed(11)
  L_new <- orig("Peak_gene_looping_L_samping")(
    args$R, args$R_sample, args$RNA_Cell_Sample_vector,
    args$TFA, args$B, args$L, args$L_state,
    args$L_prior_mean, args$L_prior_var, args$sigma_R_noise,
    args$P, args$G, args$M, args$S
  )
  expect_true(all(L_new[args$L_state == 0] == 0))
  expect_true(all(is.finite(L_new)))
})

test_that("Peak_gene_looping_L_samping is deterministic under the same seed", {
  fx <- make_synthetic_fixture(seed = 41L)
  args <- make_sampler_inputs(fx, seed = 41L)
  f <- orig("Peak_gene_looping_L_samping")
  set.seed(12)
  a <- f(args$R, args$R_sample, args$RNA_Cell_Sample_vector,
         args$TFA, args$B, args$L, args$L_state,
         args$L_prior_mean, args$L_prior_var, args$sigma_R_noise,
         args$P, args$G, args$M, args$S)
  set.seed(12)
  b <- f(args$R, args$R_sample, args$RNA_Cell_Sample_vector,
         args$TFA, args$B, args$L, args$L_state,
         args$L_prior_mean, args$L_prior_var, args$sigma_R_noise,
         args$P, args$G, args$M, args$S)
  expect_equal(a, b)
})


context("Sampler: Peak_gene_binary_looping_L_state_samping")

test_that("returns a list with L and L_state of shape (P, G)", {
  fx <- make_synthetic_fixture(seed = 42L)
  args <- make_sampler_inputs(fx, seed = 42L)
  set.seed(13)
  out <- orig("Peak_gene_binary_looping_L_state_samping")(
    args$R, args$R_sample, args$RNA_Cell_Sample_vector,
    args$TFA, args$B, args$L, args$L_state,
    args$L_prior_mean, args$L_prior_var, args$L_prior_prob,
    args$sigma_R_noise, args$P, args$G, args$M, args$S
  )
  expect_named(out, c("L", "L_state"), ignore.order = TRUE)
  expect_equal(dim(out$L),       dim(args$L))
  expect_equal(dim(out$L_state), dim(args$L_state))
})

test_that("L_state values are strictly binary in {0, 1}", {
  fx <- make_synthetic_fixture(seed = 43L)
  args <- make_sampler_inputs(fx, seed = 43L)
  set.seed(14)
  out <- orig("Peak_gene_binary_looping_L_state_samping")(
    args$R, args$R_sample, args$RNA_Cell_Sample_vector,
    args$TFA, args$B, args$L, args$L_state,
    args$L_prior_mean, args$L_prior_var, args$L_prior_prob,
    args$sigma_R_noise, args$P, args$G, args$M, args$S
  )
  expect_true(all(out$L_state %in% c(0, 1)))
})

test_that("L_state stays zero where the prior probability is zero", {
  fx <- make_synthetic_fixture(seed = 44L)
  args <- make_sampler_inputs(fx, seed = 44L)
  set.seed(15)
  out <- orig("Peak_gene_binary_looping_L_state_samping")(
    args$R, args$R_sample, args$RNA_Cell_Sample_vector,
    args$TFA, args$B, args$L, args$L_state,
    args$L_prior_mean, args$L_prior_var, args$L_prior_prob,
    args$sigma_R_noise, args$P, args$G, args$M, args$S
  )
  mask <- args$L_prior_prob == 0 & args$L_state == 0
  expect_true(all(out$L_state[mask] == 0))
  expect_true(all(out$L[mask]       == 0))
})
