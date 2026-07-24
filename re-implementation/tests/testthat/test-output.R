# Tests for MAGICAL_circuits_output.

context("MAGICAL_circuits_output")

# Build a posterior that has some entries above threshold so the writer's
# main loop actually executes.
make_posterior_with_hits <- function(fx, seed = 100L) {
  set.seed(seed)
  # Use enough iterations for at least a few probabilities to accumulate.
  MAGICAL_estimation(fx$loaded_data, fx$Candidate_circuits,
                     fx$Initial_model, iteration_num = 60L,
                     backend = "r")
}

test_that("writes a file that begins with the documented header", {
  fx <- make_synthetic_fixture(seed = 60L)
  post <- make_posterior_with_hits(fx, seed = 60L)

  out_path <- tempfile(fileext = ".txt")
  on.exit(unlink(out_path), add = TRUE)

  # Low thresholds so at least some circuits qualify on the tiny fixture.
  MAGICAL_circuits_output(
    Output_file_path = out_path,
    Candidate_circuits = fx$Candidate_circuits,
    Circuits_linkage_posterior = post,
    prob_threshold_TF_peak_binding    = 0.0,
    prob_threshold_peak_gene_looping  = 0.0
  )

  expect_true(file.exists(out_path))
  expect_gt(file.info(out_path)$size, 0)

  first_line <- readLines(out_path, n = 1L)
  expect_match(first_line, "Gene_symbol")
  expect_match(first_line, "Peak_start")
  expect_match(first_line, "Looping_prob")
})

test_that("data rows contain gene symbols from the candidate fixture", {
  fx <- make_synthetic_fixture(seed = 61L)
  post <- make_posterior_with_hits(fx, seed = 61L)

  out_path <- tempfile(fileext = ".txt")
  on.exit(unlink(out_path), add = TRUE)

  MAGICAL_circuits_output(
    Output_file_path = out_path,
    Candidate_circuits = fx$Candidate_circuits,
    Circuits_linkage_posterior = post,
    prob_threshold_TF_peak_binding    = 0.0,
    prob_threshold_peak_gene_looping  = 0.0
  )

  body <- readLines(out_path)
  # header is line 1 (plus a trailing newline turned into an extra empty line)
  data_lines <- body[-1]
  data_lines <- data_lines[nzchar(data_lines)]
  expect_gt(length(data_lines), 0)

  gene_names <- fx$Candidate_circuits$Genes$Gene_symbols
  # every data line should start with one of the candidate gene symbols
  first_tokens <- vapply(strsplit(data_lines, "\t"), `[`, character(1), 1L)
  expect_true(all(first_tokens %in% gene_names))
})

test_that("returns invisibly (function is called for its file side effect)", {
  fx <- make_synthetic_fixture(seed = 62L)
  post <- make_posterior_with_hits(fx, seed = 62L)

  out_path <- tempfile(fileext = ".txt")
  on.exit(unlink(out_path), add = TRUE)

  # Original function prints a summary; capture output to keep the test quiet
  # and assert it runs to completion without raising an error.
  expect_error(
    capture.output(
      MAGICAL_circuits_output(
        Output_file_path = out_path,
        Candidate_circuits = fx$Candidate_circuits,
        Circuits_linkage_posterior = post,
        prob_threshold_TF_peak_binding    = 0.0,
        prob_threshold_peak_gene_looping  = 0.0
      )
    ),
    NA
  )
})
