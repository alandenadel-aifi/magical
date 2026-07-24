// magical.cpp -- Rcpp/Armadillo backend for MAGICAL_estimation.
//
// Stage 2 port: replaces the stage-1 probe with a full implementation of the
// 5 Gibbs samplers plus the outer driver loop from
// R/MAGICAL_functions.R lines 560-1023. Statistical equivalence (correlation
// >= 0.90 vs. R backend) is the acceptance criterion --- the RNG stream
// differs from R's under Option B (locked in in CPP_PLAN.md).
//
// Build/load from R:
//     Rcpp::sourceCpp("re-implementation/src/magical.cpp")
//
// The pure-C++ core (functions in the anonymous namespace) is intentionally
// free of Rcpp/pybind11 dependencies so a pybind11 adapter can bind to it
// later (see PYTHON_PLAN.md stage P2).

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]
#include <RcppArmadillo.h>

#include <cmath>
#include <random>
#include <string>
#include <utility>

using arma::mat;
using arma::rowvec;
using arma::uvec;
using arma::uword;
using arma::vec;

// ---------------------------------------------------------------------------
// Pure C++/Armadillo core --- no Rcpp / pybind11 in this section.
// Callers pass Armadillo matrices + a std::mt19937_64 by reference.
// ---------------------------------------------------------------------------

namespace {

// ---- RNG helpers (all draws routed through a single mt19937_64) ----

inline double std_normal(std::mt19937_64& gen) {
  std::normal_distribution<double> N(0.0, 1.0);
  return N(gen);
}

// Mirror R idiom `aa = rnorm(k); aa[aa > 3] = 3; aa[aa < -3] = -3` --- draw
// then hard-clip at +/-3 sigma. Matches src/magical/_rng.py.
inline double trunc_std_normal(std::mt19937_64& gen) {
  double x = std_normal(gen);
  if (x > 3.0) return 3.0;
  if (x < -3.0) return -3.0;
  return x;
}

inline vec trunc_std_normal_vec(std::mt19937_64& gen, uword n) {
  vec out(n);
  for (uword i = 0; i < n; ++i) out[i] = trunc_std_normal(gen);
  return out;
}

// 1 / Gamma(shape, rate). numpy uses `scale = 1/rate`; std::gamma_distribution
// also uses (shape, scale).
inline double inv_gamma(std::mt19937_64& gen, double shape, double rate) {
  std::gamma_distribution<double> G(shape, 1.0 / rate);
  double g = G(gen);
  return 1.0 / g;
}

inline double runif01(std::mt19937_64& gen) {
  std::uniform_real_distribution<double> U(0.0, 1.0);
  return U(gen);
}

// Fisher-Yates on 0..n-1. Matches numpy's Generator.permutation semantics
// closely enough for statistical equivalence (order of iteration only).
inline uvec random_perm(std::mt19937_64& gen, uword n) {
  uvec out = arma::regspace<uvec>(0, n - 1);
  for (uword i = n; i > 1; --i) {
    std::uniform_int_distribution<uword> U(0, i - 1);
    uword j = U(gen);
    std::swap(out[i - 1], out[j]);
  }
  return out;
}

// ---- Struct: hidden TF activity state passed between Gibbs steps ----

struct TFA {
  mat T_A;       // M x N_atac
  mat T_R;       // M x N_rna
  mat T_sample;  // M x S
};

// ---- Sampler 1: TF activity ------------------------------------------------
// Port of TF_activity_T_sampling() in R/MAGICAL_functions.R.
void tf_activity_sampling(
    const mat&   A_sample,          // P x S
    const uvec&  atac_vec,          // N_atac; entries in [0, S] with 0 = none
    const uvec&  rna_vec,           // N_rna;  entries in [0, S] with 0 = none
    TFA&         tfa,
    const mat&   T_prior_mean,      // M x S
    const mat&   T_prior_var,       // M x S
    const mat&   B,                 // P x M
    const mat&   B_state,           // P x M (0/1 stored as double)
    double       sigma_A_noise,
    uword        P, uword M, uword S,
    std::mt19937_64& gen
) {
  (void) P;
  uvec TF_index = random_perm(gen, M);
  for (uword mi = 0; mi < M; ++mi) {
    uword m = TF_index[mi];
    double bstate_sum = arma::accu(B_state.col(m));
    if (bstate_sum <= 0.0) continue;

    vec    bm    = B.col(m);
    double bm_sq = arma::dot(bm, bm);

    rowvec temp_var    = (bm_sq * T_prior_var.row(m) / bstate_sum) + sigma_A_noise;
    // resid = A_sample - B * T_sample + bm * T_sample.row(m)
    mat    resid       = A_sample - B * tfa.T_sample + bm * tfa.T_sample.row(m);
    rowvec mean_T      = ((bm.t() * resid) / bstate_sum
                          + T_prior_mean.row(m) * sigma_A_noise) / temp_var;
    rowvec variance_T  = (T_prior_var.row(m) * sigma_A_noise) / temp_var;

    for (uword s = 0; s < S; ++s) {
      double sd_s = std::sqrt(std::abs(variance_T[s]));
      double aa   = trunc_std_normal(gen);
      tfa.T_sample(m, s) = aa * sd_s + mean_T[s];

      double centre = tfa.T_sample(m, s);
      // Broadcast to matching per-cell entries (T_A and T_R).
      // ATAC cells belonging to sample s+1
      uvec atac_idx = arma::find(atac_vec == (s + 1));
      for (uword i = 0; i < atac_idx.n_elem; ++i) {
        double z = std_normal(gen);
        tfa.T_A(m, atac_idx[i]) = z * sd_s + centre;
      }
      // RNA cells belonging to sample s+1
      uvec rna_idx = arma::find(rna_vec == (s + 1));
      for (uword i = 0; i < rna_idx.n_elem; ++i) {
        double z = std_normal(gen);
        tfa.T_R(m, rna_idx[i]) = z * sd_s + centre;
      }
    }
  }
}

// ---- Sampler 2: TF-peak binding weight ------------------------------------
// Port of TF_peak_binding_B_sampling().
void tf_peak_binding_sampling(
    const mat&      A_sample,
    TFA&            tfa,
    mat&            B,               // in/out
    const mat&      B_state,
    const mat&      B_prior_mean,
    const rowvec&   B_prior_var,     // 1 x M (flattened)
    double          sigma_A_noise,
    uword           P, uword M, uword S,
    std::mt19937_64& gen
) {
  uvec TF_index = random_perm(gen, M);
  for (uword mi = 0; mi < M; ++mi) {
    uword  m         = TF_index[mi];
    rowvec Tm        = tfa.T_sample.row(m);
    double Tm_sq     = arma::dot(Tm, Tm);
    double temp_var  = Tm_sq * B_prior_var[m] / S + sigma_A_noise;
    double variance_B = B_prior_var[m] * sigma_A_noise / temp_var;
    double sd_B       = std::sqrt(std::abs(variance_B));

    // resid = A_sample - B * T_sample + B.col(m) * Tm
    mat resid    = A_sample - B * tfa.T_sample + B.col(m) * Tm;
    vec mean_B   = ((resid * Tm.t()) * B_prior_var[m] / S
                    + B_prior_mean.col(m) * sigma_A_noise) / temp_var;

    vec bb       = trunc_std_normal_vec(gen, P);
    B.col(m)     = (bb * sd_B + mean_B) % B_state.col(m);
  }
}

// ---- Sampler 3: TF-peak binding STATE update ------------------------------
// Port of TF_peak_binary_binding_B_state_sampling().
void tf_peak_binary_binding_sampling(
    const mat&      A_sample,
    TFA&            tfa,
    mat&            B,
    mat&            B_state,
    const mat&      B_prior_mean,
    const rowvec&   B_prior_var,
    const mat&      B_prior_prob,
    double          sigma_A_noise,
    uword           P, uword M, uword S,
    std::mt19937_64& gen
) {
  uvec TF_index   = random_perm(gen, M);
  uvec Peak_index = random_perm(gen, P);

  for (uword fi = 0; fi < P; ++fi) {
    uword  f    = Peak_index[fi];
    // temp = B.row(f) * T_sample  (1 x S) --- computed once per f, matching R
    rowvec temp = B.row(f) * tfa.T_sample;

    for (uword mi = 0; mi < M; ++mi) {
      uword  m         = TF_index[mi];
      rowvec Tm        = tfa.T_sample.row(m);
      double Tm_sq     = arma::dot(Tm, Tm);
      double temp_var  = Tm_sq * B_prior_var[m] / S + sigma_A_noise;
      double variance_B = B_prior_var[m] * sigma_A_noise / temp_var;
      double bp        = B_prior_prob(f, m);

      if (B_state(f, m) > 0.0) {
        // residual with current-TF contribution added back
        rowvec r_row  = A_sample.row(f) - temp + B(f, m) * Tm;
        double mean_B = (arma::dot(r_row, Tm) * B_prior_var[m] / S
                         + B_prior_mean(f, m) * sigma_A_noise) / temp_var;

        double d1     = B(f, m) - mean_B;
        double d0     = -mean_B;
        double post_b1 = std::exp(-d1 * d1 / (2.0 * variance_B)) * (bp + 0.25) + 1e-6;
        double post_b0 = std::exp(-d0 * d0 / (2.0 * variance_B)) * (1.0 - bp + 0.25) + 1e-6;
        double P1 = post_b1 / (post_b1 + post_b0);
        if (!std::isfinite(P1)) P1 = 0.5;
        if (P1 < runif01(gen)) {
          B(f, m)       = 0.0;
          B_state(f, m) = 0.0;
        }
      }

      if (B_state(f, m) == 0.0 && bp > 0.0) {
        rowvec r_row  = A_sample.row(f) - temp;
        double mean_B = (arma::dot(r_row, Tm) * B_prior_var[m] / S
                         + B_prior_mean(f, m) * sigma_A_noise) / temp_var;

        double bb     = trunc_std_normal(gen);
        double B_temp = bb * std::sqrt(std::abs(variance_B)) + mean_B;

        double post_b1 = std::exp(-bb * bb / 2.0) * (bp + 0.25) + 1e-6;
        double post_b0 = std::exp(-mean_B * mean_B / (2.0 * variance_B))
                         * (1.0 - bp + 0.25) + 1e-6;
        double P1 = post_b1 / (post_b1 + post_b0);
        if (!std::isfinite(P1)) P1 = 0.5;
        if (P1 < runif01(gen)) {
          B(f, m)       = 0.0;
          B_state(f, m) = 0.0;
        } else {
          B(f, m)       = B_temp;
          B_state(f, m) = 1.0;
        }
      }
    }
  }
}

// ---- Sampler 4: peak-gene looping weight ----------------------------------
// Port of Peak_gene_looping_L_samping(). Note the missing `abs()` inside
// sqrt(variance_L) mirrors the R spelling (differs from sampler 5, per the
// Python port's comment).
void peak_gene_looping_sampling(
    const mat&    R_sample,     // G x S
    TFA&          tfa,
    const mat&    B,            // P x M
    mat&          L,            // P x G
    const mat&    L_state,      // P x G
    const mat&    L_prior_mean, // P x G
    double        L_prior_var,
    double        sigma_R_noise,
    uword         P, uword G, uword M, uword S,
    std::mt19937_64& gen
) {
  (void) M;
  mat A_estimate = B * tfa.T_sample;                 // P x S
  uvec Peak_index = random_perm(gen, P);
  for (uword fi = 0; fi < P; ++fi) {
    uword f = Peak_index[fi];
    rowvec af = A_estimate.row(f);
    double af_sq   = arma::dot(af, af);
    double temp_var = af_sq * L_prior_var / S + sigma_R_noise;
    // resid = R_sample - L.t() * A_estimate + L.row(f).t() * af  (G x S)
    mat resid = R_sample - L.t() * A_estimate + L.row(f).t() * af;
    vec mean_L = ((resid * af.t()) * L_prior_var / S
                  + L_prior_mean.row(f).t() * sigma_R_noise) / temp_var;
    double variance_L = L_prior_var * sigma_R_noise / temp_var;

    vec ll = trunc_std_normal_vec(gen, G);
    // R spelling: sqrt(variance_L) with no abs(). See magical/samplers.py.
    L.row(f) = ((ll * std::sqrt(variance_L) + mean_L) % L_state.row(f).t()).t();
  }
}

// ---- Sampler 5: peak-gene looping STATE update ----------------------------
// Port of Peak_gene_binary_looping_L_state_samping(). Note the +0.1 offsets
// (rather than the +0.25 used in sampler 3) --- matches the R state block.
void peak_gene_binary_looping_sampling(
    const mat&    R_sample,
    TFA&          tfa,
    const mat&    B,
    mat&          L,
    mat&          L_state,
    const mat&    L_prior_mean,
    double        L_prior_var,
    const mat&    L_prior_prob,
    double        sigma_R_noise,
    uword         P, uword G, uword M, uword S,
    std::mt19937_64& gen
) {
  (void) M;
  mat A_estimate = B * tfa.T_sample;
  uvec Peak_index = random_perm(gen, P);
  uvec Gene_index = random_perm(gen, G);

  for (uword gi = 0; gi < G; ++gi) {
    uword g = Gene_index[gi];
    // temp = L.col(g).t() * A_estimate  (1 x S) --- once per g, matching R
    rowvec temp = L.col(g).t() * A_estimate;

    for (uword fi = 0; fi < P; ++fi) {
      uword f = Peak_index[fi];
      rowvec af = A_estimate.row(f);
      double af_sq    = arma::dot(af, af);
      double temp_var = af_sq * L_prior_var / S + sigma_R_noise;
      double variance_L = L_prior_var * sigma_R_noise / temp_var;
      double lp = L_prior_prob(f, g);

      if (L_state(f, g) > 0.0) {
        rowvec r_row = R_sample.row(g) - temp + L(f, g) * af;
        double mean_L = (arma::dot(r_row, af) * L_prior_var / S
                         + L_prior_mean(f, g) * sigma_R_noise) / temp_var;

        double d1 = L(f, g) - mean_L;
        double d0 = -mean_L;
        double post_l1 = std::exp(-d1 * d1 / (2.0 * variance_L)) * (lp + 0.25) + 1e-6;
        double post_l0 = std::exp(-d0 * d0 / (2.0 * variance_L)) * (1.0 - lp + 0.25) + 1e-6;
        double P1 = post_l1 / (post_l1 + post_l0);
        if (!std::isfinite(P1)) P1 = 0.5;
        if (P1 < runif01(gen)) {
          L(f, g)       = 0.0;
          L_state(f, g) = 0.0;
        }
      }

      if (L_state(f, g) == 0.0 && lp > 0.0) {
        rowvec r_row = R_sample.row(g) - temp;
        double mean_L = (arma::dot(r_row, af) * L_prior_var / S
                         + L_prior_mean(f, g) * sigma_R_noise) / temp_var;

        double ll     = trunc_std_normal(gen);
        double L_temp = ll * std::sqrt(std::abs(variance_L)) + mean_L;

        // NB: +0.1 offset here, matches R state block (differs from sampler 3)
        double post_l1 = std::exp(-ll * ll / 2.0) * (lp + 0.1) + 1e-6;
        double post_l0 = std::exp(-mean_L * mean_L / (2.0 * variance_L))
                         * (1.0 - lp + 0.1) + 1e-6;
        double P1 = post_l1 / (post_l1 + post_l0);
        if (!std::isfinite(P1)) P1 = 0.5;
        if (P1 < runif01(gen)) {
          L(f, g)       = 0.0;
          L_state(f, g) = 0.0;
        } else {
          L(f, g)       = L_temp;
          L_state(f, g) = 1.0;
        }
      }
    }
  }
}

// ---- Outer driver ----------------------------------------------------------
// Port of MAGICAL_estimation() Gibbs loop (R/MAGICAL_functions.R lines
// 864-1023). The R setup block (peak_index construction etc.) is done on the
// R side; this function receives the already-prepared matrices.

struct EstimationOutput {
  mat TF_Peak_Binding_prob;   // P x M --- B_state_frq / (iter+1)
  mat Peak_Gene_Looping_prob; // P x G --- L_state_frq / (iter+1)
  mat Noise_parameters;       // iter x 2
};

EstimationOutput run_estimation(
    const mat&    A_sample,          // P x S
    const uvec&   atac_vec,          // N_atac
    const mat&    R_sample,          // G x S
    const uvec&   rna_vec,           // N_rna
    // priors + initial state
    const mat&    B_prior_mean,      // P x M
    const rowvec& B_prior_var,       // 1 x M
    const mat&    B_prior_prob,      // P x M
          mat     B,                 // P x M   in/out (copy)
          mat     B_state,           // P x M
    const mat&    L_prior_mean,      // P x G
          double  L_prior_var,       // scalar
    const mat&    L_prior_prob,      // P x G
          mat     L,                 // P x G
          mat     L_state,           // P x G
    const mat&    T_prior_mean,      // M x S
    const mat&    T_prior_var,       // M x S
    uword         N_atac,
    uword         N_rna,
    int           iteration_num,
    uint64_t      seed,
    bool          verbose
) {
  const uword P = A_sample.n_rows;
  const uword S = A_sample.n_cols;
  const uword G = R_sample.n_rows;
  const uword M = T_prior_mean.n_rows;

  std::mt19937_64 gen(seed);

  // Initial T_sample = T_prior_mean; initial T_A / T_R fanned out per sample.
  TFA tfa;
  tfa.T_sample = T_prior_mean;
  tfa.T_A.set_size(M, N_atac);  tfa.T_A.zeros();
  tfa.T_R.set_size(M, N_rna);   tfa.T_R.zeros();
  for (uword s = 1; s <= S; ++s) {
    uvec atac_idx = arma::find(atac_vec == s);
    uvec rna_idx  = arma::find(rna_vec  == s);
    for (uword m = 0; m < M; ++m) {
      double centre = T_prior_mean(m, s - 1);
      double sd_ms  = std::sqrt(std::abs(T_prior_var(m, s - 1)));
      for (uword i = 0; i < atac_idx.n_elem; ++i) {
        tfa.T_A(m, atac_idx[i]) = std_normal(gen) * sd_ms + centre;
      }
      for (uword i = 0; i < rna_idx.n_elem; ++i) {
        tfa.T_R(m, rna_idx[i]) = std_normal(gen) * sd_ms + centre;
      }
    }
  }

  // Initial noise scales.
  const double alpha_A = 1.0, beta_A = 1.0;
  const double alpha_R = 1.0, beta_R = 1.0;
  auto sigma_A_update = [&](const mat& _A, const mat& _B, const TFA& _tfa) {
    mat resid = _A - _B * _tfa.T_sample;
    double ss = arma::accu(resid % resid);
    return inv_gamma(gen, alpha_A + 0.5, (beta_A + ss) / (2.0 * P * S));
  };
  auto sigma_R_update = [&](const mat& _R, const mat& _L,
                             const mat& _B, const TFA& _tfa) {
    mat resid = _R - _L.t() * (_B * _tfa.T_sample);
    double ss = arma::accu(resid % resid);
    return inv_gamma(gen, alpha_R + 0.5, (beta_R + ss) / (2.0 * G * S));
  };
  double sigma_A_noise = sigma_A_update(A_sample, B, tfa);
  double sigma_R_noise = sigma_R_update(R_sample, L, B, tfa);

  // Frequency accumulators start from the initial 0/1 states (matches R).
  mat B_state_frq = B_state;
  mat L_state_frq = L_state;
  mat Noise_parameters(iteration_num, 2, arma::fill::zeros);

  // Latent-bug fix from the plan: round(iter/10) is 0 when iter < 6.
  int iteration_seg = std::max(1, static_cast<int>(std::round(iteration_num / 10.0)));

  if (verbose) Rcpp::Rcout << "MAGICAL integration starts ...\n";

  for (int i = 1; i <= iteration_num; ++i) {
    tf_activity_sampling(
        A_sample, atac_vec, rna_vec, tfa, T_prior_mean, T_prior_var,
        B, B_state, sigma_A_noise, P, M, S, gen);

    tf_peak_binding_sampling(
        A_sample, tfa, B, B_state, B_prior_mean, B_prior_var,
        sigma_A_noise, P, M, S, gen);

    tf_peak_binary_binding_sampling(
        A_sample, tfa, B, B_state, B_prior_mean, B_prior_var, B_prior_prob,
        sigma_A_noise, P, M, S, gen);
    B_state_frq += B_state;

    peak_gene_looping_sampling(
        R_sample, tfa, B, L, L_state, L_prior_mean, L_prior_var,
        sigma_R_noise, P, G, M, S, gen);

    peak_gene_binary_looping_sampling(
        R_sample, tfa, B, L, L_state, L_prior_mean, L_prior_var,
        L_prior_prob, sigma_R_noise, P, G, M, S, gen);
    L_state_frq += L_state;

    sigma_A_noise = sigma_A_update(A_sample, B, tfa);
    sigma_R_noise = sigma_R_update(R_sample, L, B, tfa);
    Noise_parameters(i - 1, 0) = sigma_A_noise;
    Noise_parameters(i - 1, 1) = sigma_R_noise;

    if (verbose && (i % iteration_seg == 0)) {
      Rcpp::Rcout << "MAGICAL finished " << (10 * i / iteration_seg)
                  << " percent\n";
    }
    // Also let R's interrupt handler run every so often.
    if (i % iteration_seg == 0) Rcpp::checkUserInterrupt();
  }

  EstimationOutput out;
  double denom = static_cast<double>(iteration_num) + 1.0;
  out.TF_Peak_Binding_prob   = B_state_frq / denom;
  out.Peak_Gene_Looping_prob = L_state_frq / denom;
  out.Noise_parameters       = Noise_parameters;
  return out;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Rcpp adapter --- unpacks R types and calls into the pure-C++ core above.
// ---------------------------------------------------------------------------

// Kept for the stage-1 build probe (tests/testthat/test-cpp-build.R).
// [[Rcpp::export]]
std::string magical_cpp_probe() {
  return std::string("magical.cpp loaded; armadillo ")
       + std::to_string(ARMA_VERSION_MAJOR) + "."
       + std::to_string(ARMA_VERSION_MINOR) + "."
       + std::to_string(ARMA_VERSION_PATCH);
}

// Full estimation entrypoint. Caller must pre-build the argument bundle;
// R/MAGICAL_functions.R (backend = "cpp" branch) does that.
//
// atac_vec / rna_vec are 1-based sample IDs (0 = no sample assigned).
//
// [[Rcpp::export]]
Rcpp::List magical_estimation_cpp(
    const arma::mat&    A_sample,
    const arma::uvec&   atac_vec,
    const arma::mat&    R_sample,
    const arma::uvec&   rna_vec,
    const arma::mat&    B_prior_mean,
    const arma::rowvec& B_prior_var,
    const arma::mat&    B_prior_prob,
    const arma::mat&    B,
    const arma::mat&    B_state,
    const arma::mat&    L_prior_mean,
    double              L_prior_var,
    const arma::mat&    L_prior_prob,
    const arma::mat&    L,
    const arma::mat&    L_state,
    const arma::mat&    T_prior_mean,
    const arma::mat&    T_prior_var,
    unsigned int        N_atac,
    unsigned int        N_rna,
    int                 iteration_num,
    double              seed,
    bool                verbose = false
) {
  uint64_t seed_u64 = static_cast<uint64_t>(seed);
  EstimationOutput out = run_estimation(
      A_sample, atac_vec, R_sample, rna_vec,
      B_prior_mean, B_prior_var, B_prior_prob, B, B_state,
      L_prior_mean, L_prior_var, L_prior_prob, L, L_state,
      T_prior_mean, T_prior_var,
      N_atac, N_rna, iteration_num, seed_u64, verbose);

  return Rcpp::List::create(
      Rcpp::Named("TF_Peak_Binding_prob")   = out.TF_Peak_Binding_prob,
      Rcpp::Named("Peak_Gene_Looping_prob") = out.Peak_Gene_Looping_prob,
      Rcpp::Named("Noise_parameters")       = out.Noise_parameters);
}
