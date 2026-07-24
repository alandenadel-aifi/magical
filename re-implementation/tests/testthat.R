# Standalone testthat driver for the re-implementation.
#
# Run from the repo root:
#     Rscript re-implementation/tests/testthat.R
# or interactively:
#     testthat::test_dir("re-implementation/tests/testthat")

if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("testthat is required. install.packages('testthat').")
}

# Anchor paths to the repo root regardless of caller cwd.
this_file <- tryCatch(
  normalizePath(sys.frame(1)$ofile),
  error = function(e) NULL
)
if (!is.null(this_file)) {
  repo_root <- normalizePath(file.path(dirname(this_file), "..", ".."))
  setwd(repo_root)
}

source("re-implementation/R/MAGICAL_functions.R")

testthat::test_dir("re-implementation/tests/testthat", reporter = "summary")
