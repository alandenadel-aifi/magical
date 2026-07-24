"""Port of ``Data_loading`` (R/MAGICAL_functions.R lines 8-121).

Reads the plain-text TSVs shipped in `Demo input files/` (or any dataset
with the same schema) into an in-memory :class:`~magical.types.LoadedData`
plus an extra ``motifs`` / ``TF_Peak_binding_matrix`` / ``refseq`` /
``candidate_genes`` / ``candidate_peaks`` bundle used by the circuits step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .types import LoadedData


@dataclass
class LoadedInputs:
    """Full parsed inputs including everything ``Data_loading`` returns.

    Superset of :class:`LoadedData` — carries the extra tables needed by
    ``candidate_circuits_construction_with_TAD``.
    """

    common_samples: list[str]
    candidate_genes: pd.DataFrame          # cols: Gene_symbols
    candidate_peaks: pd.DataFrame          # cols: chr, point1, point2
    scRNA_Genes: pd.DataFrame              # cols: Gene_index, Gene_symbols
    scRNA_cells: pd.DataFrame              # cols: cell_index, cell_barcode, cell_type, subject_ID, condition
    scRNA_read_count_matrix: sparse.spmatrix
    scATAC_Peaks: pd.DataFrame             # cols: Peak_index, chr, point1, point2
    scATAC_cells: pd.DataFrame             # cols: cell_index, cell_barcode, cell_type, subject_ID
    scATAC_read_count_matrix: sparse.spmatrix
    motifs: pd.DataFrame                   # cols: motif_index, name
    TF_Peak_binding_matrix: sparse.spmatrix
    refseq: pd.DataFrame                   # cols: chr, strand, start, end, Gene_symbols

    def as_loaded_data(self) -> LoadedData:
        """View as the :class:`LoadedData` used by initialization/estimation."""
        return LoadedData(
            common_samples=self.common_samples,
            scATAC_Peaks=self.scATAC_Peaks,
            scATAC_cells=self.scATAC_cells,
            scATAC_read_count_matrix=self.scATAC_read_count_matrix,
            scRNA_Genes=self.scRNA_Genes,
            scRNA_cells=self.scRNA_cells,
            scRNA_read_count_matrix=self.scRNA_read_count_matrix,
        )


def _read_triplets(path: str | Path) -> sparse.csr_matrix:
    """Read an R-style (i, j, x) triplet file into a sparse matrix.

    R uses 1-based indices; scipy uses 0-based, so we subtract 1.
    Returns CSR for efficient row-slicing downstream.
    """
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    i = arr[:, 0].astype(np.int64) - 1
    j = arr[:, 1].astype(np.int64) - 1
    x = arr[:, 2]
    n_rows = int(i.max()) + 1
    n_cols = int(j.max()) + 1
    return sparse.coo_matrix((x, (i, j)), shape=(n_rows, n_cols)).tocsr()


def data_loading(
    candidate_gene_file_path: str | Path,
    candidate_peak_file_path: str | Path,
    scRNA_readcount_file_path: str | Path,
    scRNA_gene_file_path: str | Path,
    scRNA_cellmeta_file_path: str | Path,
    scATAC_readcount_file_path: str | Path,
    scATAC_peak_file_path: str | Path,
    scATAC_cellmeta_file_path: str | Path,
    motif_mapping_file_path: str | Path,
    motif_name_file_path: str | Path,
    ref_seq_file_path: str | Path,
    *,
    verbose: bool = False,
) -> LoadedInputs:
    """Port of ``Data_loading``. Reads all input TSVs into memory."""

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ---- candidate genes -----------------------------------------------
    candidate_genes = pd.read_csv(
        candidate_gene_file_path, sep="\t", header=None, names=["Gene_symbols"], dtype=str
    )
    _log(f"{len(candidate_genes)} candidate genes are provided.")

    # ---- scRNAseq ------------------------------------------------------
    scRNA_Genes = pd.read_csv(
        scRNA_gene_file_path, sep="\t", header=None, names=["Gene_index", "Gene_symbols"],
        dtype={"Gene_index": int, "Gene_symbols": str},
    )
    scRNA_cells = pd.read_csv(
        scRNA_cellmeta_file_path, sep="\t", header=None,
        names=["cell_index", "cell_barcode", "cell_type", "subject_ID", "condition"],
        dtype={
            "cell_index": int, "cell_barcode": str, "cell_type": str,
            "subject_ID": str, "condition": str,
        },
    )
    scRNA_read_count_matrix = _read_triplets(scRNA_readcount_file_path)
    _log(
        f"scRNAseq: {len(scRNA_Genes)} Genes, {len(scRNA_cells)} cells, "
        f"{scRNA_cells['subject_ID'].nunique()} samples"
    )

    # ---- candidate peaks -----------------------------------------------
    candidate_peaks = pd.read_csv(
        candidate_peak_file_path, sep="\t", header=None,
        names=["chr", "point1", "point2"],
        dtype={"chr": str, "point1": int, "point2": int},
    )
    _log(f"{len(candidate_peaks)} candidate peaks are provided.")

    # ---- scATACseq -----------------------------------------------------
    scATAC_Peaks_raw = pd.read_csv(
        scATAC_peak_file_path, sep="\t", header=None
    )
    # R keeps only the first 4 columns: Peak_index, chr, point1, point2
    scATAC_Peaks = scATAC_Peaks_raw.iloc[:, :4].copy()
    scATAC_Peaks.columns = ["Peak_index", "chr", "point1", "point2"]
    scATAC_Peaks = scATAC_Peaks.astype(
        {"Peak_index": int, "chr": str, "point1": int, "point2": int}
    )

    scATAC_cells = pd.read_csv(
        scATAC_cellmeta_file_path, sep="\t", header=None,
        names=["cell_index", "cell_barcode", "cell_type", "subject_ID"],
        dtype={
            "cell_index": int, "cell_barcode": str, "cell_type": str, "subject_ID": str,
        },
    )
    scATAC_read_count_matrix = _read_triplets(scATAC_readcount_file_path)
    _log(
        f"scATACseq: {len(scATAC_Peaks)} Peaks, {len(scATAC_cells)} cells, "
        f"{scATAC_cells['subject_ID'].nunique()} samples"
    )

    scRNA_samples = scRNA_cells["subject_ID"].unique()
    scATAC_samples = scATAC_cells["subject_ID"].unique()
    common_samples = [s for s in scRNA_samples if s in set(scATAC_samples)]
    _log(f"Sample-paired data for {len(common_samples)} samples.")

    # ---- Motif prior ---------------------------------------------------
    motifs = pd.read_csv(
        motif_name_file_path, sep="\t", header=None, names=["motif_index", "name"],
        dtype={"motif_index": int, "name": str},
    )
    TF_Peak_binding_matrix = _read_triplets(motif_mapping_file_path)
    _log(f"{len(motifs)} motifs are screened.")

    # ---- RefSeq --------------------------------------------------------
    refseq = pd.read_csv(
        ref_seq_file_path, sep="\t", header=0,
    )
    refseq.columns = ["chr", "strand", "start", "end", "Gene_symbols"]

    return LoadedInputs(
        common_samples=common_samples,
        candidate_genes=candidate_genes,
        candidate_peaks=candidate_peaks,
        scRNA_Genes=scRNA_Genes,
        scRNA_cells=scRNA_cells,
        scRNA_read_count_matrix=scRNA_read_count_matrix,
        scATAC_Peaks=scATAC_Peaks,
        scATAC_cells=scATAC_cells,
        scATAC_read_count_matrix=scATAC_read_count_matrix,
        motifs=motifs,
        TF_Peak_binding_matrix=TF_Peak_binding_matrix,
        refseq=refseq,
    )
