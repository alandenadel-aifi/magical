#!/usr/bin/env Rscript
# Clean per-stage timing of the R pipeline on the demo dataset.
#
# Usage: Rscript bench_r_demo.R [seed] [n_iter]
# Defaults: seed=20260723, n_iter=1000
#
# Appends a row to re-implementation/benchmarks/r_demo_timings.tsv.

args      <- commandArgs(trailingOnly = TRUE)
seed      <- if (length(args) >= 1) as.integer(args[1]) else 20260723L
n_iter    <- if (length(args) >= 2) as.integer(args[2]) else 1000L
blas_path <- sessionInfo()[["BLAS"]]
blas_tag  <- if (grepl("openblas", blas_path, ignore.case = TRUE)) "openblas" else "ref_blas"
cat(sprintf("[bench_r_demo] seed=%d  n_iter=%d  blas=%s\n", seed, n_iter, blas_tag))

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
  "bench_r_demo.R"
}
this_file <- normalizePath(find_this_file())
repo_root <- normalizePath(file.path(dirname(this_file), "..", ".."))
setwd(repo_root)

source(file.path("R", "MAGICAL_functions.R"))

cat("[bench_r_demo] data_loading ...\n")
demo <- "Demo input files"
timings <- list()
t_total_start <- proc.time()[["elapsed"]]

t <- proc.time()[["elapsed"]]
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
timings$data_loading <- proc.time()[["elapsed"]] - t
cat(sprintf("[bench_r_demo]   data_loading          %8.2f s\n", timings$data_loading))
cat("[bench_r_demo] candidate_circuits ...\n")

t <- proc.time()[["elapsed"]]
Candidate_circuits <- Candidate_circuits_construction_with_TAD(
  loaded_data, file.path(demo, "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt")
)
timings$candidate_circuits <- proc.time()[["elapsed"]] - t
cat(sprintf("[bench_r_demo]   candidate_circuits    %8.2f s\n", timings$candidate_circuits))
cat("[bench_r_demo] initialization ...\n")

t <- proc.time()[["elapsed"]]
Initial_model <- MAGICAL_initialization(loaded_data, Candidate_circuits)
timings$initialization <- proc.time()[["elapsed"]] - t
cat(sprintf("[bench_r_demo]   initialization        %8.2f s\n", timings$initialization))
cat(sprintf("[bench_r_demo] estimation (%d iter) ...\n", n_iter))

set.seed(seed)
t <- proc.time()[["elapsed"]]
Result <- MAGICAL_estimation(loaded_data, Candidate_circuits, Initial_model,
                             iteration_num = n_iter)
timings[[paste0("estimation_", n_iter, "iter")]] <- proc.time()[["elapsed"]] - t
cat(sprintf("[bench_r_demo]   estimation            %8.2f s\n", timings[[paste0("estimation_", n_iter, "iter")]]))

timings$total <- proc.time()[["elapsed"]] - t_total_start
cat(sprintf("[bench_r_demo]   total                 %8.2f s\n", timings$total))

out_dir <- file.path("re-implementation", "benchmarks")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_file <- file.path(out_dir, "r_demo_timings.tsv")

row <- data.frame(
  seed   = seed,
  n_iter = n_iter,
  blas   = blas_tag,
  data_loading_seconds       = round(timings$data_loading, 3),
  candidate_circuits_seconds = round(timings$candidate_circuits, 3),
  initialization_seconds     = round(timings$initialization, 3),
  estimation_seconds         = round(timings[[paste0("estimation_", n_iter, "iter")]], 3),
  total_seconds              = round(timings$total, 3),
  stringsAsFactors = FALSE
)
append_mode <- file.exists(out_file)
write.table(row, out_file, sep = "\t", quote = FALSE, row.names = FALSE,
            col.names = !append_mode, append = append_mode)
cat("Wrote", out_file, "\n")
print(row)

# ---- Save posterior snapshots (same layout as bench_cpp_demo.R) ----------
snap_dir <- file.path("tests",
                      sprintf("snapshots_demo_R_%diter_%d", n_iter, seed))
dir.create(snap_dir, showWarnings = FALSE, recursive = TRUE)
write_mat <- function(m, name) {
  write.table(as.matrix(m), file.path(snap_dir, paste0(name, ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
}
write_mat(Result$TF_Peak_Binding_prob,   "TF_Peak_Binding_prob")
write_mat(Result$Peak_Gene_Looping_prob, "Peak_Gene_Looping_prob")
write_mat(Result$Noise_parameters,       "Noise_parameters")
writeLines(as.character(n_iter), file.path(snap_dir, "n_iter.txt"))
writeLines(as.character(seed),   file.path(snap_dir, "seed.txt"))
write.table(data.frame(name = Candidate_circuits$TFs$name),
            file.path(snap_dir, "TFs.tsv"),   sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(Peak_index = Candidate_circuits$Peaks$Peak_index),
            file.path(snap_dir, "Peaks.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(Gene_symbols = Candidate_circuits$Genes$Gene_symbols),
            file.path(snap_dir, "Genes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("Wrote snapshots to", snap_dir, "\n")
