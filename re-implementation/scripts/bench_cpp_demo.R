# Benchmark the cpp backend on the shipped demo dataset at n_iter=1000.
# Writes timings to re-implementation/benchmarks/cpp_demo_timings.tsv and
# saves the posterior matrices to tests/snapshots_demo_Cpp_1000iter_<seed>/
# so plot_variability.py can pick them up.
#
# Usage:
#   Rscript re-implementation/scripts/bench_cpp_demo.R [seed] [n_iter]
#
# Both args are optional; defaults are seed=20260723, n_iter=1000.

args <- commandArgs(trailingOnly = TRUE)
seed   <- if (length(args) >= 1) as.integer(args[1]) else 20260723L
n_iter <- if (length(args) >= 2) as.integer(args[2]) else 1000L

# Locate the repo root: this script lives in re-implementation/scripts/.
script_path <- sub("^--file=", "",
                   commandArgs(trailingOnly = FALSE)[
                     grep("^--file=", commandArgs(trailingOnly = FALSE))
                   ])
repo_root <- if (length(script_path) && nchar(script_path)) {
  normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  getwd()
}
setwd(repo_root)

message(sprintf("[bench_cpp_demo] seed=%d  n_iter=%d  cwd=%s",
                seed, n_iter, getwd()))

suppressPackageStartupMessages({
  library(Matrix)
  library(dplyr)
})

source("re-implementation/R/MAGICAL_functions.R")
suppressWarnings(magical_load_cpp())
stopifnot(exists("magical_estimation_cpp", mode = "function"))

# ---- load demo inputs (same as tutorial.R / gen_demo_parity_snapshots.R) --
demo <- "Demo input files"
t0 <- Sys.time()
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
t_load <- as.numeric(Sys.time() - t0, units = "secs")

t0 <- Sys.time()
Candidate_circuits <- Candidate_circuits_construction_with_TAD(
  loaded_data,
  file.path(demo, "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt")
)
t_circ <- as.numeric(Sys.time() - t0, units = "secs")

t0 <- Sys.time()
Initial_model <- MAGICAL_initialization(loaded_data, Candidate_circuits)
t_init <- as.numeric(Sys.time() - t0, units = "secs")

message(sprintf("[bench_cpp_demo] setup done  (load=%.1fs circuits=%.1fs init=%.1fs)",
                t_load, t_circ, t_init))

# ---- run cpp estimation ---------------------------------------------------
t0 <- Sys.time()
out <- MAGICAL_estimation(loaded_data, Candidate_circuits, Initial_model,
                          iteration_num = n_iter, backend = "cpp",
                          seed = seed, verbose = FALSE)
t_est <- as.numeric(Sys.time() - t0, units = "secs")
t_total <- t_load + t_circ + t_init + t_est
message(sprintf("[bench_cpp_demo] estimation=%.2fs  total=%.2fs", t_est, t_total))

# ---- save timings ---------------------------------------------------------
bench_dir <- "re-implementation/benchmarks"
dir.create(bench_dir, showWarnings = FALSE, recursive = TRUE)
timings_file <- file.path(bench_dir, "cpp_demo_timings.tsv")
row <- data.frame(
  seed          = seed,
  n_iter        = n_iter,
  data_loading_seconds       = round(t_load, 3),
  candidate_circuits_seconds = round(t_circ, 3),
  initialization_seconds     = round(t_init, 3),
  estimation_seconds         = round(t_est, 3),
  total_seconds              = round(t_total, 3),
  stringsAsFactors = FALSE
)
append <- file.exists(timings_file)
write.table(row, timings_file, sep = "\t", quote = FALSE, row.names = FALSE,
            col.names = !append, append = append)
message(sprintf("[bench_cpp_demo] wrote %s", timings_file))

# ---- save snapshots (same layout as gen_demo_parity_snapshots.R) ---------
snap_dir <- sprintf("tests/snapshots_demo_Cpp_%diter_%d", n_iter, seed)
dir.create(snap_dir, showWarnings = FALSE, recursive = TRUE)
write_mat <- function(m, name) {
  path <- file.path(snap_dir, paste0(name, ".tsv"))
  write.table(as.matrix(m), path, sep = "\t", quote = FALSE,
              row.names = FALSE, col.names = FALSE)
}
write_mat(out$TF_Peak_Binding_prob,   "TF_Peak_Binding_prob")
write_mat(out$Peak_Gene_Looping_prob, "Peak_Gene_Looping_prob")
write_mat(out$Noise_parameters,       "Noise_parameters")
writeLines(as.character(n_iter), file.path(snap_dir, "n_iter.txt"))
writeLines(as.character(seed),   file.path(snap_dir, "seed.txt"))
write.table(data.frame(TF = Candidate_circuits$TFs[, 1]),   file.path(snap_dir, "TFs.tsv"),   sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(Peak = Candidate_circuits$Peaks$Peak_index), file.path(snap_dir, "Peaks.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(Gene = Candidate_circuits$Genes$Gene_symbols), file.path(snap_dir, "Genes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
message(sprintf("[bench_cpp_demo] wrote snapshots to %s", snap_dir))

message("=== bench_cpp_demo DONE ===")
