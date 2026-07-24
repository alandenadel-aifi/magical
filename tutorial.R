source('R/MAGICAL_functions.R')

library(Matrix)
library(dplyr)

# pre-selected candidate genes and peaks for the cell type
Candidate_gene_file_path = 'Demo input files/Cell type candidate genes.txt'
Candidate_peak_file_path = 'Demo input files/Cell type candidate peaks.txt'

# filtered scRNA data of the cell type
scRNA_readcount_file_path = 'Demo input files/Cell type scRNA read count.txt'
scRNA_gene_file_path = 'Demo input files/scRNA genes.txt'
scRNA_cellmeta_file_path = 'Demo input files/Cell type scRNA cell meta.txt'

# filtered scATAC data of the cell type
scATAC_readcount_file_path = 'Demo input files/Cell type scATAC read count.txt'
scATAC_peak_file_path = 'Demo input files/scATAC peaks.txt'
scATAC_cellmeta_file_path = 'Demo input files/Cell type scATAC cell meta.txt'

# TF motif prior on all ATAC peaks
Motif_mapping_file_path = 'Demo input files/Motif mapping prior.txt'
Motif_name_file_path = 'Demo input files/Motifs.txt'

# TAD prior
TAD_file_path = 'Demo input files/RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt'
distance_control=5e5

# Refseq file for transcription starting site extraction
Ref_seq_file_path = 'Demo input files/hg38_Refseq.txt'

loaded_data <- Data_loading(Candidate_gene_file_path, Candidate_peak_file_path,
                            scRNA_readcount_file_path, scRNA_gene_file_path, scRNA_cellmeta_file_path,
                            scATAC_readcount_file_path, scATAC_peak_file_path, scATAC_cellmeta_file_path,
                            Motif_mapping_file_path, Motif_name_file_path, Ref_seq_file_path)

#Candidate circuits construction with TAD
Candidate_circuits <- Candidate_circuits_construction_with_TAD(loaded_data, TAD_file_path)

#Candidate circuits construction without TAD
#Candidate_circuits <- Candidate_circuits_construction_without_TAD(loaded_data, distance_control)

# Model parameter estimation with parallel chains
library(parallel)

n_chains <- 4        # adjust to available cores (detectCores() to check)
iteration_num <- 1000

# L'Ecuyer-CMRG gives independent RNG streams across forked processes
RNGkind("L'Ecuyer-CMRG")
set.seed(42)

start_time <- proc.time()
chain_results <- mclapply(seq_len(n_chains), function(chain_id) {
  set.seed(1000 + chain_id)
  chain_init <- MAGICAL_initialization(loaded_data, Candidate_circuits)
  MAGICAL_estimation(loaded_data, Candidate_circuits, chain_init, iteration_num = iteration_num)
}, mc.cores = n_chains)
elapsed_time <- proc.time() - start_time
print('Elapsed time for MAGICAL estimation (parallel chains):')
print(elapsed_time)

# Pool posterior probabilities by averaging across chains
Circuits_linkage_posterior <- list(
  TF_Peak_Binding_prob  = Reduce('+', lapply(chain_results, `[[`, 'TF_Peak_Binding_prob'))  / n_chains,
  Peak_Gene_Looping_prob = Reduce('+', lapply(chain_results, `[[`, 'Peak_Gene_Looping_prob')) / n_chains,
  Noise_parameters      = Reduce('+', lapply(chain_results, `[[`, 'Noise_parameters'))      / n_chains
)

MAGICAL_circuits_output(Output_file_path = 'MAGICAL_selected_regulatory_circuits.txt', 
                        Candidate_circuits, Circuits_linkage_posterior)

