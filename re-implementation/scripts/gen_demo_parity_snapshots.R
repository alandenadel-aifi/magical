#!/usr/bin/env Rscript
# Generate demo-scale R snapshots for cross-language parity tests.
#
# Runs the FULL R pipeline (Data_loading -> Candidate_circuits ->
# MAGICAL_initialization -> MAGICAL_estimation) on the shipped
# `Demo input files/`, then writes the intermediate and final artifacts to
# `tests/snapshots_demo[_<suffix>]/*.tsv` for the Python numpy port to
# compare against.
#
# Usage:
#   Rscript gen_demo_parity_snapshots.R [seed] [out_suffix] [n_iter]
#     seed        : integer seed for MAGICAL_estimation (default 20260723)
#     out_suffix  : output dir tests/snapshots_demo[_<suffix>] (default "")
#     n_iter      : Gibbs iterations (default 50; use 1000 for paper defaults)

suppressPackageStartupMessages({
  library(Matrix)
  library(dplyr)
})

find_this_file <- function() {
  cargs <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", cargs, value = TRUE)
  if (length(m) > 0) return(sub("^--file=", "", m[1]))
  f <- try(sys.frame(1L)$ofile, silent = TRUE)
  if (!inherits(f, "try-error") && !is.null(f)) return(f)
  "gen_demo_parity_snapshots.R"
}
this_file <- normalizePath(find_this_file())
repo_root <- normalizePath(file.path(dirname(this_file), "..", ".."))
setwd(repo_root)

source(file.path("R", "MAGICAL_functions.R"))

args <- commandArgs(trailingOnly = TRUE)
seed <- if (length(args) >= 1) as.integer(args[1]) else 20260723L
out_suffix <- if (length(args) >= 2) args[2] else ""
n_iter <- if (length(args) >= 3) as.integer(args[3]) else 50L
out_dir <- if (nchar(out_suffix) > 0) {
  file.path("tests", paste0("snapshots_demo_", out_suffix))
} else {
  file.path("tests", "snapshots_demo")
}
cat("R snapshot run with seed=", seed, ", n_iter=", n_iter,
    ", out_dir=", out_dir, "\n", sep = "")

demo <- "Demo input files"
loaded_data <- Data_loading(
  file.path(demo, "Cell type candidate genes.txt"),
  file.path(demo, "Cell type candidate peaks.txt"),
  file.path(demo, "Cell type scRNA read count.txt"),
  file.path(demo, "scRNA genes.txt"),
  file.path(demo, "Cell type scRNA cell meta.txt"),
  file.path(demo, "Cell type scATAC read count.txt"),
  file.path(demo, "scATAC peaks.txt"),
  file.path(demo, "Cell type scATAC cell meta.txt"),
  file.path(demo, "Motif mapping prior.txt"),
  file.path(demo, "Motifs.txt"),
  file.path(demo, "hg38_Refseq.txt")
)
Candidate_circuits <- Candidate_circuits_construction_with_TAD(
  loaded_data, file.path(demo, "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt")
)
Initial_model <- MAGICAL_initialization(loaded_data, Candidate_circuits)

set.seed(seed)
t0 <- proc.time()[["elapsed"]]
Result <- MAGICAL_estimation(loaded_data, Candidate_circuits, Initial_model,
                             iteration_num = n_iter)
estimation_seconds <- proc.time()[["elapsed"]] - t0

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_mat <- function(m, name) {
  write.table(as.matrix(m), file.path(out_dir, paste0(name, ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
}
write_df <- function(df, name) {
  write.table(df, file.path(out_dir, paste0(name, ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
}

# Candidate_circuits (needed so Python can locate identical row/col ordering)
write_df(Candidate_circuits$TFs, "TFs")
write_df(Candidate_circuits$Peaks, "Peaks")
write_df(Candidate_circuits$Genes, "Genes")
write_mat(Candidate_circuits$TF_log2Count,   "TF_log2Count")
write_mat(Candidate_circuits$Peak_log2Count, "Peak_log2Count")
write_mat(Candidate_circuits$Gene_log2Count, "Gene_log2Count")
write_mat(as.matrix(Candidate_circuits$TF_Peak_Binding),   "TF_Peak_Binding")
write_mat(as.matrix(Candidate_circuits$Peak_Gene_looping), "Peak_Gene_looping")

# Initialization (deterministic)
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

# Estimation (RNG-dependent)
write_mat(Result$TF_Peak_Binding_prob,   "TF_Peak_Binding_prob")
write_mat(Result$Peak_Gene_Looping_prob, "Peak_Gene_Looping_prob")
write_mat(Result$Noise_parameters,       "Noise_parameters")

writeLines(as.character(n_iter), file.path(out_dir, "n_iter.tsv"))
writeLines(as.character(seed),   file.path(out_dir, "seed.tsv"))
writeLines(loaded_data$Common_samples, file.path(out_dir, "Common_samples.tsv"))
writeLines(sprintf("estimation_seconds\t%.4f", estimation_seconds),
           file.path(out_dir, "timings.tsv"))

cat("Wrote demo snapshots to", out_dir,
    "  (estimation:", sprintf("%.1fs", estimation_seconds), ")\n")
