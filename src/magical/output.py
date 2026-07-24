"""Port of ``MAGICAL_circuits_output``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import CandidateCircuits, EstimationResult


def magical_circuits_output(
    output_file_path: str | Path,
    candidate_circuits: CandidateCircuits,
    circuits_linkage_posterior: EstimationResult,
    prob_threshold_TF_peak_binding: float = 0.8,
    prob_threshold_peak_gene_looping: float = 0.95,
) -> None:
    """Write the selected regulatory circuits to a text file.

    Format matches [R/MAGICAL_functions.R](../R/MAGICAL_functions.R) L1034+.
    """
    path = Path(output_file_path)

    L_prob = circuits_linkage_posterior.Peak_Gene_Looping_prob
    B_prob = circuits_linkage_posterior.TF_Peak_Binding_prob

    # np.argwhere walks row-major; R's ``arr.ind = TRUE`` walks column-major.
    # We match column-major so the output ordering matches the R version.
    mask = L_prob > prob_threshold_peak_gene_looping
    cols, rows = np.nonzero(mask.T)
    peak_gene_index = np.column_stack([rows, cols])

    tfs_df = candidate_circuits.TFs
    genes_df = candidate_circuits.Genes
    peaks_df = candidate_circuits.Peaks

    tf_vector = np.zeros(tfs_df.shape[0], dtype=int)

    with path.open("w") as fh:
        fh.write(
            "Gene_symbol\tGene_chr\tGene_TSS\tPeak_chr\tPeak_start\tPeak_end\t"
            "Looping_prob\tTFs(binding prob)\n\n"
        )

        for peak_i, gene_i in peak_gene_index:
            tf_probs = B_prob[peak_i, :]
            order = np.argsort(tf_probs)[::-1]
            selected = order[tf_probs[order] > prob_threshold_TF_peak_binding]
            if selected.size == 0:
                fh.write("\n")
                continue

            fh.write(
                "\t".join(
                    [
                        str(genes_df.iloc[gene_i]["Gene_symbols"]),
                        str(genes_df.iloc[gene_i]["chr"]),
                        str(genes_df.iloc[gene_i]["TSS"]),
                        str(peaks_df.iloc[peak_i]["chr"]),
                        str(peaks_df.iloc[peak_i]["point1"]),
                        str(peaks_df.iloc[peak_i]["point2"]),
                        str(L_prob[peak_i, gene_i]),
                    ]
                )
            )
            fh.write("\t")
            for j in selected:
                # Second column of TFs (0-based col 1) is the TF name
                tf_name = tfs_df.iloc[j, 1]
                fh.write(f"{tf_name} ,( {float(tf_probs[j])} ), sep ")
                tf_vector[j] += 1
            fh.write("\n")
