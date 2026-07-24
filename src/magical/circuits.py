"""Port of ``Candidate_circuits_construction_with_TAD`` (R lines 124-306)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .io import LoadedInputs
from .types import CandidateCircuits


def candidate_circuits_construction_with_TAD(
    loaded: LoadedInputs, TAD_file_path: str | Path, *, verbose: bool = False
) -> CandidateCircuits:
    """Numpy port of the TAD-based candidate-circuit construction."""

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    common_samples = list(loaded.common_samples)
    candidate_genes = loaded.candidate_genes.copy()
    candidate_peaks = loaded.candidate_peaks.copy()
    scRNA_Genes = loaded.scRNA_Genes
    scRNA_cells = loaded.scRNA_cells
    scRNA_read = loaded.scRNA_read_count_matrix.tocsr()
    scATAC_Peaks = loaded.scATAC_Peaks
    scATAC_cells = loaded.scATAC_cells
    scATAC_read = loaded.scATAC_read_count_matrix.tocsr()
    motifs = loaded.motifs
    TF_Peak_binding = loaded.TF_Peak_binding_matrix.tocsr()
    refseq = loaded.refseq

    # -- TAD prior --------------------------------------------------------
    TAD = pd.read_csv(
        TAD_file_path, sep="\t", header=None,
        names=["chr", "left_boundary", "right_boundary"],
        dtype={"chr": str, "left_boundary": int, "right_boundary": int},
    )
    _log(f"{len(TAD)} TAD segments provided")

    # -- TF-Peak candidate filtering -------------------------------------
    # inner_join(scATAC_Peaks, Candidate_Peaks, c("chr", "point1", "point2"))
    # dplyr::inner_join keeps left-table column order + rows in the order
    # they appear in the *left* table (scATAC_Peaks) that match anything in
    # the right table.
    merged = scATAC_Peaks.merge(
        candidate_peaks, on=["chr", "point1", "point2"], how="inner"
    )
    candidate_peaks = merged.reset_index(drop=True)

    # Map back to row indices in scATAC_Peaks / TF_Peak_binding
    peak_index_lookup = pd.Series(
        np.arange(len(scATAC_Peaks)), index=scATAC_Peaks["Peak_index"].values
    )
    row_idx = peak_index_lookup.reindex(candidate_peaks["Peak_index"].values).values
    row_idx = np.asarray(row_idx, dtype=np.int64)
    candidate_tf_peak = TF_Peak_binding[row_idx, :]

    tf_pct_cand = np.asarray(candidate_tf_peak.sum(axis=0)).ravel() / len(candidate_peaks)
    tf_num = np.asarray(candidate_tf_peak.sum(axis=0)).ravel()
    tf_pct_all = np.asarray(TF_Peak_binding.sum(axis=0)).ravel() / TF_Peak_binding.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        tf_enrichment_FC = np.where(tf_pct_all > 0, tf_pct_cand / tf_pct_all, 0.0)
    tf_keep = np.flatnonzero(
        (tf_pct_cand > 0.05) & (tf_num > 30) & (tf_enrichment_FC > 0.8)
    )
    if tf_keep.size == 0:
        raise RuntimeError(
            "Too few Peaks with TF binding sites. MAGICAL not applicable to this cell type!"
        )
    candidate_tfs = motifs.iloc[tf_keep].reset_index(drop=True)
    candidate_tf_peak = candidate_tf_peak[:, tf_keep]

    peak_keep = np.flatnonzero(np.asarray(candidate_tf_peak.sum(axis=1)).ravel() > 0)
    candidate_peaks = candidate_peaks.iloc[peak_keep].reset_index(drop=True)
    candidate_tf_peak = candidate_tf_peak[peak_keep, :]

    # -- Gene TSS from RefSeq --------------------------------------------
    gene_symbols = sorted(
        set(candidate_genes["Gene_symbols"])
        & set(refseq["Gene_symbols"])
        & set(scRNA_Genes["Gene_symbols"])
    )
    # NB: R uses intersect() which preserves first-argument order and de-dupes.
    order_in_candidate = pd.Index(candidate_genes["Gene_symbols"]).unique()
    seen = set(gene_symbols)
    gene_symbols = [g for g in order_in_candidate if g in seen]
    # Now filter by second intersect (Refseq order preserved from previous step)
    gene_symbols = [g for g in gene_symbols if g in set(refseq["Gene_symbols"])]
    gene_symbols = [g for g in gene_symbols if g in set(scRNA_Genes["Gene_symbols"])]

    # Build gene TSS
    refseq_by_gene = refseq.groupby("Gene_symbols")
    chrs: list[str] = []
    tsss: list[int] = []
    for g in gene_symbols:
        grp = refseq_by_gene.get_group(g)
        chrs.append(str(grp.iloc[0]["chr"]))
        if grp.iloc[0]["strand"] == "+":
            tsss.append(int(grp["start"].min()))
        else:
            tsss.append(int(grp["end"].max()))
    candidate_genes = pd.DataFrame(
        {"Gene_symbols": gene_symbols, "chr": chrs, "TSS": tsss}
    )

    # -- Peak-Gene looping (TAD * distance <1M) --------------------------
    n_peak = len(candidate_peaks)
    n_gene = len(candidate_genes)

    tad_by_chr = TAD.groupby("chr")
    peaks_by_chr = candidate_peaks.groupby("chr")
    genes_by_chr = candidate_genes.groupby("chr")

    rows_tad: list[int] = []
    cols_tad: list[int] = []
    for chrom, tads in tad_by_chr:
        if chrom not in peaks_by_chr.groups or chrom not in genes_by_chr.groups:
            continue
        peaks_c = peaks_by_chr.get_group(chrom)
        genes_c = genes_by_chr.get_group(chrom)
        peak_p1 = peaks_c["point1"].to_numpy()
        peak_p2 = peaks_c["point2"].to_numpy()
        gene_tss = genes_c["TSS"].to_numpy()
        peak_ids = peaks_c.index.to_numpy()
        gene_ids = genes_c.index.to_numpy()
        for lb, rb in zip(tads["left_boundary"].to_numpy(), tads["right_boundary"].to_numpy()):
            p_mask = (peak_p1 > lb) & (peak_p2 < rb)
            g_mask = (gene_tss > lb) & (gene_tss < rb)
            if p_mask.any() and g_mask.any():
                pi = peak_ids[p_mask]
                gi = gene_ids[g_mask]
                rr, cc = np.meshgrid(pi, gi, indexing="ij")
                rows_tad.extend(rr.ravel().tolist())
                cols_tad.extend(cc.ravel().tolist())
    tad_matrix = sparse.coo_matrix(
        (
            np.ones(len(rows_tad), dtype=np.int8),
            (np.asarray(rows_tad, dtype=np.int64), np.asarray(cols_tad, dtype=np.int64)),
        ),
        shape=(n_peak, n_gene),
    ).tocsr()
    tad_matrix.data[:] = 1  # de-dup: any positive means "inside a TAD"
    tad_matrix.eliminate_zeros()

    # Distance <1M
    rows_d: list[int] = []
    cols_d: list[int] = []
    for chrom in genes_by_chr.groups:
        if chrom not in peaks_by_chr.groups:
            continue
        peaks_c = peaks_by_chr.get_group(chrom)
        genes_c = genes_by_chr.get_group(chrom)
        peak_mid = (peaks_c["point1"].to_numpy() + peaks_c["point2"].to_numpy()) / 2.0
        peak_ids = peaks_c.index.to_numpy()
        for g_idx, tss in zip(genes_c.index.to_numpy(), genes_c["TSS"].to_numpy()):
            mask = np.abs(peak_mid - tss) < 1e6
            if mask.any():
                rows_d.extend(peak_ids[mask].tolist())
                cols_d.extend([int(g_idx)] * int(mask.sum()))
    distance_matrix = sparse.coo_matrix(
        (
            np.ones(len(rows_d), dtype=np.int8),
            (np.asarray(rows_d, dtype=np.int64), np.asarray(cols_d, dtype=np.int64)),
        ),
        shape=(n_peak, n_gene),
    ).tocsr()
    distance_matrix.data[:] = 1
    distance_matrix.eliminate_zeros()

    peak_gene_looping = tad_matrix.multiply(distance_matrix).tocsr()

    peak_keep = np.flatnonzero(np.asarray(peak_gene_looping.sum(axis=1)).ravel() > 0)
    gene_keep = np.flatnonzero(np.asarray(peak_gene_looping.sum(axis=0)).ravel() > 0)
    peak_gene_looping = peak_gene_looping[peak_keep, :][:, gene_keep]
    candidate_peaks = candidate_peaks.iloc[peak_keep].reset_index(drop=True)
    candidate_genes = candidate_genes.iloc[gene_keep].reset_index(drop=True)

    tf_keep = np.flatnonzero(
        np.asarray(candidate_tf_peak[peak_keep, :].sum(axis=0)).ravel() > 0
    )
    candidate_tfs = candidate_tfs.iloc[tf_keep].reset_index(drop=True)
    candidate_tf_peak = candidate_tf_peak[peak_keep, :][:, tf_keep]

    # -- Pseudobulk ATAC -------------------------------------------------
    subject_id_atac = scATAC_cells["subject_ID"].values
    n_atac_features = scATAC_read.shape[0]
    atac_count = np.zeros((n_atac_features, len(common_samples)))
    for s, name in enumerate(common_samples):
        cols = np.flatnonzero(subject_id_atac == name)
        if cols.size == 0:
            continue
        atac_count[:, s] = np.asarray(scATAC_read[:, cols].sum(axis=1)).ravel()
    scale = 5e6 / (atac_count.sum(axis=0) + 1)
    atac_count = atac_count * scale[np.newaxis, :]
    atac_log2 = np.log2(atac_count + 1)

    peak_id_to_row = pd.Series(
        np.arange(len(scATAC_Peaks)), index=scATAC_Peaks["Peak_index"].values
    )
    row_idx = peak_id_to_row.reindex(candidate_peaks["Peak_index"].values).to_numpy()
    row_idx = np.asarray(row_idx, dtype=np.int64)
    peak_log2 = atac_log2[row_idx, :]
    peak_log2 = peak_log2 - peak_log2.mean(axis=1, keepdims=True)

    # -- Pseudobulk RNA --------------------------------------------------
    subject_id_rna = scRNA_cells["subject_ID"].values
    n_rna_features = scRNA_read.shape[0]
    rna_count = np.zeros((n_rna_features, len(common_samples)))
    for s, name in enumerate(common_samples):
        cols = np.flatnonzero(subject_id_rna == name)
        if cols.size == 0:
            continue
        rna_count[:, s] = np.asarray(scRNA_read[:, cols].sum(axis=1)).ravel()
    scale = 5e6 / (rna_count.sum(axis=0) + 1)
    rna_count = rna_count * scale[np.newaxis, :]
    rna_log2 = np.log2(rna_count + 1)

    gene_id_to_row = pd.Series(
        np.arange(len(scRNA_Genes)), index=scRNA_Genes["Gene_symbols"].values
    )
    row_idx = gene_id_to_row.reindex(candidate_genes["Gene_symbols"].values).to_numpy()
    row_idx = np.asarray(row_idx, dtype=np.int64)
    gene_log2 = rna_log2[row_idx, :]
    gene_log2 = gene_log2 - gene_log2.mean(axis=1, keepdims=True)

    # -- Restrict TFs to those expressed in scRNA ------------------------
    tf_names_expr = list(candidate_tfs["name"])
    scrna_gene_set = set(scRNA_Genes["Gene_symbols"])
    selected_tf_names = [t for t in tf_names_expr if t in scrna_gene_set]

    # Reorder candidate_tfs / candidate_tf_peak by selected_tf_names
    name_to_col = {n: c for c, n in enumerate(candidate_tfs["name"])}
    tf_reorder = np.array([name_to_col[n] for n in selected_tf_names], dtype=np.int64)
    candidate_tfs = candidate_tfs.iloc[tf_reorder].reset_index(drop=True)
    candidate_tf_peak = candidate_tf_peak[:, tf_reorder]

    # Look up their RNA log2 vector
    tf_rna_row = gene_id_to_row.reindex(selected_tf_names).to_numpy().astype(np.int64)
    tf_log2 = rna_log2[tf_rna_row, :]

    # Drop TFs whose rowSums == 0
    tf_keep = np.flatnonzero(tf_log2.sum(axis=1) > 0)
    candidate_tfs = candidate_tfs.iloc[tf_keep].reset_index(drop=True)
    candidate_tf_peak = candidate_tf_peak[:, tf_keep]
    tf_log2 = tf_log2[tf_keep, :]
    tf_log2 = tf_log2 - tf_log2.mean(axis=1, keepdims=True)

    # -- Final filtering -------------------------------------------------
    peak_keep = np.flatnonzero(
        (np.asarray(peak_gene_looping.sum(axis=1)).ravel() > 0)
        & (np.asarray(candidate_tf_peak.sum(axis=1)).ravel() > 0)
    )
    gene_keep = np.flatnonzero(
        np.asarray(peak_gene_looping[peak_keep, :].sum(axis=0)).ravel() > 0
    )
    tf_keep = np.flatnonzero(
        np.asarray(candidate_tf_peak[peak_keep, :].sum(axis=0)).ravel() > 0
    )

    candidate_tf_peak = candidate_tf_peak[peak_keep, :][:, tf_keep]
    peak_gene_looping = peak_gene_looping[peak_keep, :][:, gene_keep]
    candidate_tfs = candidate_tfs.iloc[tf_keep].reset_index(drop=True)
    tf_log2 = tf_log2[tf_keep, :]
    candidate_peaks = candidate_peaks.iloc[peak_keep].reset_index(drop=True)
    peak_log2 = peak_log2[peak_keep, :]
    candidate_genes = candidate_genes.iloc[gene_keep].reset_index(drop=True)
    gene_log2 = gene_log2[gene_keep, :]

    return CandidateCircuits(
        TFs=candidate_tfs,
        TF_log2Count=tf_log2,
        Peaks=candidate_peaks,
        Peak_log2Count=peak_log2,
        Genes=candidate_genes,
        Gene_log2Count=gene_log2,
        TF_Peak_Binding=candidate_tf_peak,
        Peak_Gene_looping=peak_gene_looping,
    )
