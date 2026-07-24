#!/usr/bin/env Rscript
# Clean per-stage timing of the R pipeline on the demo dataset.
#
# Writes re-implementation/benchmarks/r_demo_timings.tsv.

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

t <- proc.time()[["elapsed"]]
Candidate_circuits <- Candidate_circuits_construction_with_TAD(
  loaded_data, file.path(demo, "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt")
)
timings$candidate_circuits <- proc.time()[["elapsed"]] - t

t <- proc.time()[["elapsed"]]
Initial_model <- MAGICAL_initialization(loaded_data, Candidate_circuits)
timings$initialization <- proc.time()[["elapsed"]] - t

set.seed(20260723L)
t <- proc.time()[["elapsed"]]
Result <- MAGICAL_estimation(loaded_data, Candidate_circuits, Initial_model,
                             iteration_num = 50L)
timings$estimation_50iter <- proc.time()[["elapsed"]] - t

timings$total <- proc.time()[["elapsed"]] - t_total_start

out_dir <- file.path("re-implementation", "benchmarks")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_file <- file.path(out_dir, "r_demo_timings.tsv")

df <- data.frame(
  stage = names(timings),
  wall_seconds = round(unlist(timings), 4),
  stringsAsFactors = FALSE
)
write.table(df, out_file, sep = "\t", quote = FALSE, row.names = FALSE)
cat("Wrote", out_file, "\n")
print(df)
