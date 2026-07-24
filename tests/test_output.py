"""Output-file tests mirroring ``test-output.R``."""

from __future__ import annotations

import numpy as np

from magical import estimation, magical_circuits_output


def test_output_file_is_created(fixture, tmp_path) -> None:
    result = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=6, backend="numpy", seed=0,
    )
    # Force at least one circuit through by lowering thresholds
    out = tmp_path / "circuits.txt"
    magical_circuits_output(
        out, fixture.candidate_circuits, result,
        prob_threshold_TF_peak_binding=0.0,
        prob_threshold_peak_gene_looping=0.0,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_output_file_has_expected_header(fixture, tmp_path) -> None:
    result = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=6, backend="numpy", seed=0,
    )
    out = tmp_path / "circuits.txt"
    magical_circuits_output(
        out, fixture.candidate_circuits, result,
        prob_threshold_TF_peak_binding=0.0,
        prob_threshold_peak_gene_looping=0.0,
    )
    header = out.read_text().splitlines()[0]
    for expected in ("Gene_symbol", "Gene_chr", "Peak_start", "Looping_prob"):
        assert expected in header


def test_output_lines_reference_gene_and_tf_tokens(fixture, tmp_path) -> None:
    result = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=6, backend="numpy", seed=0,
    )
    out = tmp_path / "circuits.txt"
    magical_circuits_output(
        out, fixture.candidate_circuits, result,
        prob_threshold_TF_peak_binding=0.0,
        prob_threshold_peak_gene_looping=0.0,
    )
    text = out.read_text()
    # At least one gene name and one TF name should appear
    gene_hits = sum(
        1 for g in fixture.candidate_circuits.Genes["Gene_symbols"] if g in text
    )
    tf_hits = sum(
        1 for t in fixture.candidate_circuits.TFs.iloc[:, 1] if t in text
    )
    assert gene_hits > 0
    assert tf_hits > 0
