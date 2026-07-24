#!/usr/bin/env Rscript
# Generate R-side snapshots for cross-language parity tests.
#
# Writes a synthetic fixture (identical structure to helper-synthetic.R but
# with numeric values chosen for portability) plus the outputs of
# MAGICAL_initialization and MAGICAL_estimation into
#   tests/snapshots/*.tsv
# so the Python numpy port can compare against them.

suppressPackageStartupMessages({
  library(Matrix)
})

# Locate repo root (this script lives at re-implementation/scripts/)
# Prefer commandArgs (works under Rscript); fall back to sys.frame (interactive).
find_this_file <- function() {
  cargs <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", cargs, value = TRUE)
  if (length(m) > 0) return(sub("^--file=", "", m[1]))
  f <- try(sys.frame(1L)$ofile, silent = TRUE)
  if (!inherits(f, "try-error") && !is.null(f)) return(f)
  "gen_parity_snapshots.R"
}
this_file <- normalizePath(find_this_file())
repo_root <- normalizePath(file.path(dirname(this_file), "..", ".."))
setwd(repo_root)

source(file.path("R", "MAGICAL_functions.R"))

# ---------------------------------------------------------------------------
# Build a fixture with fully-specified values (no RNG), so Python can
# reconstruct the exact same inputs from the saved matrices.
# ---------------------------------------------------------------------------
set.seed(20260723L)

S <- 6L; M <- 5L; P <- 20L; G <- 10L

sample_names <- paste0("S", seq_len(S))
tf_names     <- paste0("TF",   seq_len(M))
peak_ids     <- paste0("peak", seq_len(P))
gene_names   <- paste0("gene", seq_len(G))

TF_log2Count   <- matrix(rnorm(M * S, 5, 1), nrow = M, dimnames = list(tf_names, sample_names))
Peak_log2Count <- matrix(rnorm(P * S, 3, 1), nrow = P, dimnames = list(peak_ids, sample_names))
Gene_log2Count <- matrix(rnorm(G * S, 4, 1), nrow = G, dimnames = list(gene_names, sample_names))

TF_Peak_Binding_dense <- matrix(rbinom(P * M, 1, 0.35), nrow = P, dimnames = list(peak_ids, tf_names))
for (m in seq_len(M)) if (sum(TF_Peak_Binding_dense[, m]) == 0) TF_Peak_Binding_dense[1, m] <- 1
TF_Peak_Binding <- Matrix(TF_Peak_Binding_dense, sparse = TRUE)

Peak_Gene_looping_dense <- matrix(rbinom(P * G, 1, 0.30), nrow = P, dimnames = list(peak_ids, gene_names))
for (g in seq_len(G)) if (sum(Peak_Gene_looping_dense[, g]) == 0) Peak_Gene_looping_dense[1, g] <- 1
Peak_Gene_looping <- Matrix(Peak_Gene_looping_dense, sparse = TRUE)

# scATAC / scRNA cell metadata (8 cells per sample per modality)
build_cellmeta <- function(prefix, cells_per_sample) {
  rows <- list()
  for (s in seq_len(S)) {
    for (k in seq_len(cells_per_sample)) {
      rows[[length(rows) + 1]] <- data.frame(
        cell_index   = 0L,
        cell_barcode = paste0(prefix, "_", sample_names[s], "_", k),
        cell_type    = "T",
        subject_ID   = sample_names[s],
        stringsAsFactors = FALSE
      )
    }
  }
  meta <- do.call(rbind, rows)
  meta$cell_index <- seq_len(nrow(meta))
  meta
}
cells_per_sample <- 8L
scATAC_cells <- build_cellmeta("atac", cells_per_sample)
scRNA_cells  <- build_cellmeta("rna",  cells_per_sample)
scRNA_cells$condition <- "cond"

build_readcount <- function(n_features, cells, mean_count, feature_names) {
  Nc <- nrow(cells)
  counts <- matrix(rpois(n_features * Nc, mean_count), nrow = n_features, ncol = Nc)
  rownames(counts) <- feature_names
  as(counts, "TsparseMatrix")
}

scATAC_Peaks <- data.frame(
  Peak_index = peak_ids,
  chr        = "chr1",
  point1     = seq_len(P) * 1000L,
  point2     = seq_len(P) * 1000L + 500L,
  stringsAsFactors = FALSE
)
scATAC_read_count_matrix <- build_readcount(P, scATAC_cells, 2, peak_ids)

rna_feature_names <- unique(c(gene_names, tf_names))
scRNA_Genes <- data.frame(
  Gene_index   = seq_along(rna_feature_names),
  Gene_symbols = rna_feature_names,
  stringsAsFactors = FALSE
)
scRNA_read_count_matrix <- build_readcount(length(rna_feature_names), scRNA_cells, 3, rna_feature_names)

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
  TFs               = data.frame(TF_id = tf_names, TF_name = tf_names, stringsAsFactors = FALSE),
  TF_log2Count      = TF_log2Count,
  Peaks             = data.frame(Peak_index = peak_ids,
                                  chr = "chr1",
                                  point1 = seq_len(P) * 1000L,
                                  point2 = seq_len(P) * 1000L + 500L,
                                  stringsAsFactors = FALSE),
  Peak_log2Count    = Peak_log2Count,
  Genes             = data.frame(Gene_symbols = gene_names,
                                  chr = "chr1",
                                  TSS = seq_len(G) * 5000L,
                                  stringsAsFactors = FALSE),
  Gene_log2Count    = Gene_log2Count,
  TF_Peak_Binding   = TF_Peak_Binding,
  Peak_Gene_looping = Peak_Gene_looping
)

# ---------------------------------------------------------------------------
# Run R-side initialization + estimation
# ---------------------------------------------------------------------------
Initial_model <- MAGICAL_initialization(loaded_data, Candidate_circuits)

set.seed(20260723L)
n_iter <- 50L
# NOTE: original R crashes for iteration_num < 6 due to `iteration_seg == 0`
Result <- MAGICAL_estimation(loaded_data, Candidate_circuits, Initial_model,
                             iteration_num = n_iter)

# ---------------------------------------------------------------------------
# Write snapshot files
# ---------------------------------------------------------------------------
out_dir <- file.path("tests", "snapshots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_mat <- function(m, name) {
  write.table(as.matrix(m), file.path(out_dir, paste0(name, ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
}

# Inputs
write_mat(TF_log2Count,   "TF_log2Count")
write_mat(Peak_log2Count, "Peak_log2Count")
write_mat(Gene_log2Count, "Gene_log2Count")
write_mat(TF_Peak_Binding_dense,  "TF_Peak_Binding")
write_mat(Peak_Gene_looping_dense, "Peak_Gene_looping")

# Initialization outputs (deterministic, no RNG)
write_mat(Initial_model$T_prior, "T_prior")
write_mat(Initial_model$T_mean,  "T_mean")
write_mat(Initial_model$T_var,   "T_var")
write_mat(Initial_model$B_prior, "B_prior")
write_mat(Initial_model$B_mean,  "B_mean")
write_mat(Initial_model$B_var,   "B_var")
write_mat(Initial_model$B_prob,  "B_prob")
write_mat(Initial_model$L_prior, "L_prior")
write_mat(Initial_model$L_mean,  "L_mean")
writeLines(as.character(Initial_model$L_var),
           file.path(out_dir, "L_var.tsv"))
write_mat(Initial_model$L_prob,  "L_prob")

# Estimation outputs (RNG-dependent — Python matches by correlation only)
write_mat(Result$TF_Peak_Binding_prob,  "TF_Peak_Binding_prob")
write_mat(Result$Peak_Gene_Looping_prob, "Peak_Gene_Looping_prob")
write_mat(Result$Noise_parameters,       "Noise_parameters")

writeLines(as.character(n_iter), file.path(out_dir, "n_iter.tsv"))

cat("Wrote snapshots to", out_dir, "\n")
