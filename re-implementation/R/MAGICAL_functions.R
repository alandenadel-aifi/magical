#***************************************************************************
#  MAGICAL re-implementation (Rcpp/Armadillo backend)
#
#  Public API is a drop-in replacement for R/MAGICAL_functions.R.
#  Currently STAGE 1: the R side is a thin wrapper that delegates to the
#  original R implementation. Subsequent stages will replace the body of
#  MAGICAL_estimation() with a call into src/magical.cpp while keeping the
#  wrapper signature and return-list layout identical.
#
#  Usage (from the repo root, same convention as tutorial.R):
#     source('re-implementation/R/MAGICAL_functions.R')
#***************************************************************************

# --- locate and load the original implementation into a private env --------
.magical_original_env <- new.env(parent = globalenv())

local({
  candidates <- c(
    "R/MAGICAL_functions.R",
    "../R/MAGICAL_functions.R",
    "../../R/MAGICAL_functions.R"
  )
  hit <- Find(file.exists, candidates)
  if (is.null(hit)) {
    stop("re-implementation: could not locate original R/MAGICAL_functions.R; ",
         "source this file from the repo root.")
  }
  sys.source(normalizePath(hit), envir = .magical_original_env)
})

# --- re-export the pieces we do not plan to port --------------------------
# (I/O + one-off construction; the bottleneck is entirely in the MCMC loop.)
Data_loading                            <- .magical_original_env$Data_loading
Candidate_circuits_construction_with_TAD    <- .magical_original_env$Candidate_circuits_construction_with_TAD
Candidate_circuits_construction_without_TAD <- .magical_original_env$Candidate_circuits_construction_without_TAD
MAGICAL_initialization                  <- .magical_original_env$MAGICAL_initialization
MAGICAL_circuits_output                 <- .magical_original_env$MAGICAL_circuits_output

# --- optional Rcpp backend (loaded lazily; stage 1 = no backend yet) ------
.magical_cpp_loaded <- FALSE

magical_load_cpp <- function(force = FALSE) {
  if (.magical_cpp_loaded && !force) return(invisible(TRUE))
  src_candidates <- c(
    "re-implementation/src/magical.cpp",
    "src/magical.cpp",
    "../src/magical.cpp",
    "../../src/magical.cpp",
    "../../../src/magical.cpp"
  )
  hit <- Find(file.exists, src_candidates)
  if (is.null(hit)) stop("magical.cpp not found on any known relative path.")
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("Rcpp is required for the C++ backend. install.packages('Rcpp').")
  }
  Rcpp::sourceCpp(normalizePath(hit))
  assign(".magical_cpp_loaded", TRUE, envir = topenv())
  invisible(TRUE)
}

# --- MAGICAL_estimation: same signature; dispatches to R or cpp backend ---
MAGICAL_estimation <- function(loaded_data, Candidate_circuits, Initial_model,
                               iteration_num, backend = c("r", "cpp"),
                               seed = NULL, verbose = FALSE) {
  backend <- match.arg(backend)

  if (backend == "r") {
    return(.magical_original_env$MAGICAL_estimation(
      loaded_data, Candidate_circuits, Initial_model, iteration_num
    ))
  }

  # backend == "cpp" ----------------------------------------------------
  magical_load_cpp()
  if (!exists("magical_estimation_cpp", mode = "function")) {
    stop("magical_estimation_cpp() not exported by the compiled backend.")
  }

  Common_samples <- loaded_data$Common_samples
  S <- length(Common_samples)

  # --- ATAC data setup (mirrors R MAGICAL_estimation head + Python driver)
  Peak_index <- match(Candidate_circuits$Peaks$Peak_index,
                      loaded_data$scATAC_Peaks$Peak_index)
  A_sample <- as.matrix(Candidate_circuits$Peak_log2Count)
  atac_vec <- integer(nrow(loaded_data$scATAC_cells))
  for (s in seq_len(S)) {
    atac_vec[loaded_data$scATAC_cells$subject_ID == Common_samples[s]] <- s
  }

  # --- RNA data setup
  Gene_index <- match(Candidate_circuits$Genes$Gene_symbols,
                      loaded_data$scRNA_Genes$Gene_symbols)
  R_sample <- as.matrix(Candidate_circuits$Gene_log2Count)
  rna_vec <- integer(nrow(loaded_data$scRNA_cells))
  for (s in seq_len(S)) {
    rna_vec[loaded_data$scRNA_cells$subject_ID == Common_samples[s]] <- s
  }

  # --- Priors + initial state (dense matrices as C++ expects)
  B_prior_mean <- as.matrix(Initial_model$B_mean)
  B_prior_var  <- as.numeric(as.matrix(Initial_model$B_var))    # 1 x M -> vec
  B_prior_prob <- as.matrix(Initial_model$B_prob)
  B_init       <- as.matrix(Initial_model$B_prior)
  B_state_init <- as.matrix(Candidate_circuits$TF_Peak_Binding) * 1.0

  L_prior_mean <- as.matrix(Initial_model$L_mean)
  L_prior_var  <- as.numeric(Initial_model$L_var)               # scalar
  L_prior_prob <- as.matrix(Initial_model$L_prob)
  L_init       <- as.matrix(Initial_model$L_prior)
  L_state_init <- as.matrix(Candidate_circuits$Peak_Gene_looping) * 1.0

  T_prior_mean <- as.matrix(Initial_model$T_mean)
  T_prior_var  <- as.matrix(Initial_model$T_var)

  seed_val <- if (is.null(seed)) as.numeric(sample.int(.Machine$integer.max, 1))
              else as.numeric(seed)

  out <- magical_estimation_cpp(
    A_sample     = A_sample,
    atac_vec     = as.integer(atac_vec),
    R_sample     = R_sample,
    rna_vec      = as.integer(rna_vec),
    B_prior_mean = B_prior_mean,
    B_prior_var  = as.numeric(B_prior_var),
    B_prior_prob = B_prior_prob,
    B            = B_init,
    B_state      = B_state_init,
    L_prior_mean = L_prior_mean,
    L_prior_var  = L_prior_var,
    L_prior_prob = L_prior_prob,
    L            = L_init,
    L_state      = L_state_init,
    T_prior_mean = T_prior_mean,
    T_prior_var  = T_prior_var,
    N_atac       = as.integer(length(atac_vec)),
    N_rna        = as.integer(length(rna_vec)),
    iteration_num = as.integer(iteration_num),
    seed         = seed_val,
    verbose      = isTRUE(verbose)
  )

  # Restore R rownames/colnames on the returned matrices so downstream
  # MAGICAL_circuits_output works unchanged.
  dimnames(out$TF_Peak_Binding_prob)   <- dimnames(B_state_init)
  dimnames(out$Peak_Gene_Looping_prob) <- dimnames(L_state_init)
  colnames(out$Noise_parameters)       <- c("sigma_A_noise", "sigma_R_noise")
  out
}
