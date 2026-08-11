# Full-dataset integration test on the shipped demo inputs.
#
# Opt-in only: this runs the full Data_loading + estimation pipeline, which
# takes minutes on the demo data. Enable with:
#     MAGICAL_RUN_SLOW_TESTS=1 Rscript re-implementation/tests/testthat.R

context("Demo-dataset integration (opt-in slow)")

test_that("r backend runs to completion on demo data and returns valid probs", {
  skip_if(Sys.getenv("MAGICAL_RUN_SLOW_TESTS") != "1",
          "set MAGICAL_RUN_SLOW_TESTS=1 to enable")

  demo_dir <- testthat::test_path("../../../Demo input files")
  skip_if_not(dir.exists(demo_dir),
              paste("demo data not found at", demo_dir))

  loaded_data <- Data_loading(
    file.path(demo_dir, "Cell type candidate genes.txt"),
    file.path(demo_dir, "Cell type candidate peaks.txt"),
    file.path(demo_dir, "Cell type scRNA read count.txt"),
    file.path(demo_dir, "scRNA genes.txt"),
    file.path(demo_dir, "Cell type scRNA cell meta.txt"),
    file.path(demo_dir, "Cell type scATAC read count.txt"),
    file.path(demo_dir, "scATAC peaks.txt"),
    file.path(demo_dir, "Cell type scATAC cell meta.txt"),
    file.path(demo_dir, "Motif mapping prior.txt"),
    file.path(demo_dir, "Motifs.txt"),
    file.path(demo_dir, "hg38_Refseq.txt")
  )

  Candidate_circuits <- Candidate_circuits_construction_with_TAD(
    loaded_data,
    file.path(demo_dir, "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt")
  )
  Initial_model <- MAGICAL_initialization(loaded_data, Candidate_circuits)

  # Small iteration count to keep the opt-in test tractable.
  n_iter <- as.integer(Sys.getenv("MAGICAL_DEMO_ITERS", 50L))

  set.seed(20260722)
  out <- MAGICAL_estimation(loaded_data, Candidate_circuits, Initial_model,
                            iteration_num = n_iter, backend = "r")

  expect_named(out, c("TF_Peak_Binding_prob",
                      "Peak_Gene_Looping_prob",
                      "Noise_parameters"),
               ignore.order = TRUE)
  expect_true(all(is.finite(as.matrix(out$TF_Peak_Binding_prob))))
  expect_true(all(is.finite(as.matrix(out$Peak_Gene_Looping_prob))))
})

test_that("cpp backend agrees with r backend on demo data", {
  skip_if(Sys.getenv("MAGICAL_RUN_SLOW_TESTS") != "1",
          "set MAGICAL_RUN_SLOW_TESTS=1 to enable")
  skip_if(Sys.getenv("MAGICAL_TEST_CPP_BUILD") != "1",
          "set MAGICAL_TEST_CPP_BUILD=1 to enable")
  skip_if_not_installed("Rcpp")
  skip_if_not_installed("RcppArmadillo")

  ok <- tryCatch({ magical_load_cpp(); TRUE },
                 error = function(e) { message(conditionMessage(e)); FALSE })
  skip_if_not(ok, "magical.cpp failed to compile")
  skip_if_not(exists("magical_estimation_cpp", mode = "function"),
              "cpp entrypoint not exported")

  demo_dir <- testthat::test_path("../../../Demo input files")
  skip_if_not(dir.exists(demo_dir),
              paste("demo data not found at", demo_dir))

  loaded_data <- Data_loading(
    file.path(demo_dir, "Cell type candidate genes.txt"),
    file.path(demo_dir, "Cell type candidate peaks.txt"),
    file.path(demo_dir, "Cell type scRNA read count.txt"),
    file.path(demo_dir, "scRNA genes.txt"),
    file.path(demo_dir, "Cell type scRNA cell meta.txt"),
    file.path(demo_dir, "Cell type scATAC read count.txt"),
    file.path(demo_dir, "scATAC peaks.txt"),
    file.path(demo_dir, "Cell type scATAC cell meta.txt"),
    file.path(demo_dir, "Motif mapping prior.txt"),
    file.path(demo_dir, "Motifs.txt"),
    file.path(demo_dir, "hg38_Refseq.txt")
  )

  Candidate_circuits <- Candidate_circuits_construction_with_TAD(
    loaded_data,
    file.path(demo_dir, "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt")
  )
  Initial_model <- MAGICAL_initialization(loaded_data, Candidate_circuits)

  n_iter <- as.integer(Sys.getenv("MAGICAL_DEMO_ITERS", 50L))

  set.seed(20260722)
  run_r   <- MAGICAL_estimation(loaded_data, Candidate_circuits, Initial_model,
                                iteration_num = n_iter, backend = "r")
  run_cpp <- MAGICAL_estimation(loaded_data, Candidate_circuits, Initial_model,
                                iteration_num = n_iter, backend = "cpp",
                                seed = 20260722L)

  binding_corr <- cor(as.numeric(as.matrix(run_r$TF_Peak_Binding_prob)),
                      as.numeric(as.matrix(run_cpp$TF_Peak_Binding_prob)))
  looping_corr <- cor(as.numeric(as.matrix(run_r$Peak_Gene_Looping_prob)),
                      as.numeric(as.matrix(run_cpp$Peak_Gene_Looping_prob)))

  expect_true(!is.na(binding_corr) && binding_corr >= 0.90,
              info = sprintf("binding corr = %.3f", binding_corr))
  expect_true(!is.na(looping_corr) && looping_corr >= 0.90,
              info = sprintf("looping corr = %.3f", looping_corr))
})
