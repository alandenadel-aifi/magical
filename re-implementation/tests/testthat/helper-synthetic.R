# Synthetic MAGICAL fixture.
#
# Builds a small but structurally-valid `loaded_data`, `Candidate_circuits`,
# and `Initial_model` so tests can exercise MAGICAL_estimation without
# touching the 1 GB demo files.
#
# Dimensions are intentionally small (single-digit cells per sample) so a full
# MCMC run finishes in well under a second.

suppressPackageStartupMessages({
  library(Matrix)
})

make_synthetic_fixture <- function(
    seed = 1L,
    n_samples = 6L,
    n_tfs     = 5L,
    n_peaks   = 20L,
    n_genes   = 10L,
    cells_per_sample_atac = 8L,
    cells_per_sample_rna  = 8L,
    binding_density = 0.35,
    looping_density = 0.30
) {
  set.seed(seed)

  S <- n_samples
  M <- n_tfs
  P <- n_peaks
  G <- n_genes

  sample_names <- paste0("S", seq_len(S))
  tf_names     <- paste0("TF",   seq_len(M))
  peak_ids     <- paste0("peak", seq_len(P))
  gene_names   <- paste0("gene", seq_len(G))

  # per-sample log2 count summaries (dense M/P/G x S)
  TF_log2Count   <- matrix(rnorm(M * S, mean = 5, sd = 1), nrow = M, ncol = S,
                            dimnames = list(tf_names,   sample_names))
  Peak_log2Count <- matrix(rnorm(P * S, mean = 3, sd = 1), nrow = P, ncol = S,
                            dimnames = list(peak_ids,   sample_names))
  Gene_log2Count <- matrix(rnorm(G * S, mean = 4, sd = 1), nrow = G, ncol = S,
                            dimnames = list(gene_names, sample_names))

  # sparse candidate binding / looping matrices (binary 0/1)
  TF_Peak_Binding <- Matrix(
    rbinom(P * M, 1, binding_density),
    nrow = P, ncol = M,
    dimnames = list(peak_ids, tf_names),
    sparse = TRUE
  )
  # guarantee every TF has at least one candidate peak
  for (m in seq_len(M)) if (sum(TF_Peak_Binding[, m]) == 0) TF_Peak_Binding[1, m] <- 1

  Peak_Gene_looping <- Matrix(
    rbinom(P * G, 1, looping_density),
    nrow = P, ncol = G,
    dimnames = list(peak_ids, gene_names),
    sparse = TRUE
  )
  for (g in seq_len(G)) if (sum(Peak_Gene_looping[, g]) == 0) Peak_Gene_looping[1, g] <- 1

  # per-cell scATAC / scRNA read counts (sparse, cells as columns)
  build_cellmeta <- function(prefix, cells_per_sample) {
    lst <- lapply(seq_len(S), function(s) {
      k <- cells_per_sample
      data.frame(
        cell_index  = NA_integer_,  # filled after concat
        cell_barcode = paste0(prefix, "_", sample_names[s], "_", seq_len(k)),
        cell_type   = "T",
        subject_ID  = sample_names[s],
        stringsAsFactors = FALSE
      )
    })
    meta <- do.call(rbind, lst)
    meta$cell_index <- seq_len(nrow(meta))
    meta
  }

  scATAC_cells <- build_cellmeta("atac", cells_per_sample_atac)
  scRNA_cells  <- build_cellmeta("rna",  cells_per_sample_rna)
  # scRNA meta in the real Data_loading also has a 'condition' column
  scRNA_cells$condition <- "cond"

  build_readcount <- function(n_features, cells, mean_count) {
    Nc <- nrow(cells)
    counts <- matrix(rpois(n_features * Nc, lambda = mean_count),
                     nrow = n_features, ncol = Nc)
    as(counts, "TsparseMatrix")
  }

  # scATAC features must include *at least* every candidate peak.
  # We make the full scATAC peak universe equal to the candidate peaks
  # (extras aren't required for the estimation path).
  scATAC_Peaks <- data.frame(
    Peak_index = peak_ids,
    chr        = "chr1",
    point1     = seq_len(P) * 1000L,
    point2     = seq_len(P) * 1000L + 500L,
    stringsAsFactors = FALSE
  )
  scATAC_read_count_matrix <- build_readcount(P, scATAC_cells, mean_count = 2)
  rownames(scATAC_read_count_matrix) <- peak_ids

  # scRNA features must include every candidate gene *and* every candidate TF.
  # We union gene_names and tf_names, keeping order consistent.
  rna_feature_names <- unique(c(gene_names, tf_names))
  scRNA_Genes <- data.frame(
    Gene_index   = seq_along(rna_feature_names),
    Gene_symbols = rna_feature_names,
    stringsAsFactors = FALSE
  )
  scRNA_read_count_matrix <- build_readcount(length(rna_feature_names),
                                             scRNA_cells, mean_count = 3)
  rownames(scRNA_read_count_matrix) <- rna_feature_names

  loaded_data <- list(
    Common_samples           = sample_names,
    scATAC_Peaks             = scATAC_Peaks,
    scATAC_cells             = scATAC_cells,
    scATAC_read_count_matrix = scATAC_read_count_matrix,
    scRNA_Genes              = scRNA_Genes,
    scRNA_cells              = scRNA_cells,
    scRNA_read_count_matrix  = scRNA_read_count_matrix
  )

  Candidate_circuits <- list(
    # MAGICAL_circuits_output reads TFs[, 2] so keep 2+ columns.
    TFs               = data.frame(TF_id      = tf_names,
                                    TF_name    = tf_names,
                                    stringsAsFactors = FALSE),
    TF_log2Count      = TF_log2Count,
    # output also reads Peaks$chr / point1 / point2
    Peaks             = data.frame(Peak_index = peak_ids,
                                    chr        = "chr1",
                                    point1     = seq_len(P) * 1000L,
                                    point2     = seq_len(P) * 1000L + 500L,
                                    stringsAsFactors = FALSE),
    Peak_log2Count    = Peak_log2Count,
    # output also reads Genes$chr / TSS
    Genes             = data.frame(Gene_symbols = gene_names,
                                    chr          = "chr1",
                                    TSS          = seq_len(G) * 5000L,
                                    stringsAsFactors = FALSE),
    Gene_log2Count    = Gene_log2Count,
    TF_Peak_Binding   = TF_Peak_Binding,
    Peak_Gene_looping = Peak_Gene_looping
  )

  # Build Initial_model with the real initializer so the fixture matches
  # exactly what MAGICAL_estimation would see downstream.
  Initial_model <- MAGICAL_initialization(loaded_data, Candidate_circuits)

  list(
    loaded_data        = loaded_data,
    Candidate_circuits = Candidate_circuits,
    Initial_model      = Initial_model
  )
}

# Build the exact argument bundle each of the six samplers expects.
# Mirrors the setup block at the top of MAGICAL_estimation() so unit tests
# can call individual samplers with realistic inputs.
make_sampler_inputs <- function(fx, seed = 123L) {
  set.seed(seed)

  loaded_data        <- fx$loaded_data
  Candidate_circuits <- fx$Candidate_circuits
  Initial_model      <- fx$Initial_model

  Common_samples <- loaded_data$Common_samples
  S <- length(Common_samples)

  Peak_index <- match(Candidate_circuits$Peaks$Peak_index,
                      loaded_data$scATAC_Peaks$Peak_index)
  A <- loaded_data$scATAC_read_count_matrix[Peak_index, , drop = FALSE]
  ATAC_Cell_Sample_vector <- matrix(0, nrow = 1,
                                    ncol = nrow(loaded_data$scATAC_cells))
  for (s in seq_len(S)) {
    ATAC_Cell_Sample_vector[which(
      loaded_data$scATAC_cells$subject_ID == Common_samples[s]
    )] <- s
  }
  A_sample <- Candidate_circuits$Peak_log2Count
  P <- length(Peak_index)

  Gene_index <- match(Candidate_circuits$Genes$Gene_symbols,
                      loaded_data$scRNA_Genes$Gene_symbols)
  R <- loaded_data$scRNA_read_count_matrix[Gene_index, , drop = FALSE]
  RNA_Cell_Sample_vector <- matrix(0, nrow = 1,
                                   ncol = nrow(loaded_data$scRNA_cells))
  for (s in seq_len(S)) {
    RNA_Cell_Sample_vector[1, which(
      loaded_data$scRNA_cells$subject_ID == Common_samples[s]
    )] <- s
  }
  R_sample <- Candidate_circuits$Gene_log2Count
  G <- length(Gene_index)

  B_prior_mean <- Initial_model$B_mean
  B_prior_var  <- Initial_model$B_var
  B_prior_prob <- Initial_model$B_prob
  B            <- Initial_model$B_prior
  B_state      <- as.matrix(Candidate_circuits$TF_Peak_Binding)

  L_prior_mean <- Initial_model$L_mean
  L_prior_var  <- Initial_model$L_var
  L_prior_prob <- Initial_model$L_prob
  L            <- Initial_model$L_prior
  L_state      <- as.matrix(Candidate_circuits$Peak_Gene_looping)

  T_prior_mean <- Initial_model$T_mean
  T_prior_var  <- Initial_model$T_var
  T_sample     <- T_prior_mean
  M            <- nrow(T_sample)

  T_A <- matrix(0, nrow = M, ncol = nrow(loaded_data$scATAC_cells))
  for (s in seq_len(S)) {
    idx <- which(ATAC_Cell_Sample_vector == s)
    for (m in seq_len(M)) {
      T_A[m, idx] <- rnorm(length(idx),
                           mean = T_prior_mean[m, s],
                           sd   = sqrt(abs(T_prior_var[m, s])))
    }
  }
  T_R <- matrix(0, nrow = M, ncol = nrow(loaded_data$scRNA_cells))
  for (s in seq_len(S)) {
    idx <- which(RNA_Cell_Sample_vector == s)
    for (m in seq_len(M)) {
      T_R[m, idx] <- rnorm(length(idx),
                           mean = T_prior_mean[m, s],
                           sd   = sqrt(abs(T_prior_var[m, s])))
    }
  }
  TFA <- list(T_A = T_A, T_R = T_R, T_sample = T_sample)

  alpha_A <- 1; beta_A <- 1
  sigma_A_noise <- 1 / rgamma(
    1, shape = alpha_A + 1/2,
    rate  = (beta_A + sum((A_sample - B %*% TFA$T_sample)^2)) / (2 * P * S)
  )
  alpha_R <- 1; beta_R <- 1
  sigma_R_noise <- 1 / rgamma(
    1, shape = alpha_R + 1/2,
    rate  = (beta_R + sum((R_sample - t(L) %*% (B %*% TFA$T_sample))^2)) /
              (2 * G * S)
  )

  list(
    A = A, A_sample = A_sample, ATAC_Cell_Sample_vector = ATAC_Cell_Sample_vector,
    R = R, R_sample = R_sample, RNA_Cell_Sample_vector = RNA_Cell_Sample_vector,
    TFA = TFA,
    T_prior_mean = T_prior_mean, T_prior_var = T_prior_var,
    B = B, B_state = B_state,
    B_prior_mean = B_prior_mean, B_prior_var = B_prior_var,
    B_prior_prob = B_prior_prob,
    L = L, L_state = L_state,
    L_prior_mean = L_prior_mean, L_prior_var = L_prior_var,
    L_prior_prob = L_prior_prob,
    sigma_A_noise = sigma_A_noise, sigma_R_noise = sigma_R_noise,
    P = P, G = G, M = M, S = S
  )
}

# Accessor for the original samplers (which live in the private env loaded by
# the wrapper). Tests use this to unit-test individual samplers without going
# through MAGICAL_estimation.
orig <- function(fn_name) {
  env <- get(".magical_original_env", envir = globalenv())
  env[[fn_name]]
}
