# Full-dataset integration test on the shipped demo inputs.
#
# Opt-in only: this runs the full Data_loading + estimation pipeline, which
# takes minutes on the demo data. Enable with:
#     MAGICAL_RUN_SLOW_TESTS=1 Rscript re-implementation/tests/testthat.R

context("Demo-dataset integration (opt-in slow)")

test_that("r backend runs to completion on demo data and returns valid probs", {
  skip_if(Sys.getenv("MAGICAL_RUN_SLOW_TESTS") != "1",
          "set MAGICAL_RUN_SLOW_TESTS=1 to enable")

  demo_dir <- "Demo input files"
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
  skip_if_not(exists("magical_estimation_cpp"),
              "cpp backend not implemented yet (stage 1 skeleton)")

  # Placeholder for the demo-scale equivalence check; fleshed out in stage 2
  # once the cpp backend exists and we know a representative iteration count.
  skip("filled in during stage 2")
})
