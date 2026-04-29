#!/usr/bin/env python
"""Generate tiny golden outputs for ReCoN explore workflows.

This script is intentionally manual: run it only when the explore algorithms
or their expected output contract changes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from recon.explore.recon import Celltype, Multicell, multicell_targets


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"


def tiny_networks():
    """Return a deterministic, biologically shaped miniature ReCoN input set."""
    grn = pd.DataFrame({
        "source": ["TF1", "TF1", "TF2", "TF2", "TF1"],
        "target": ["LIG1", "GENE1", "REC1", "GENE2", "REC2"],
        "weight": [1.0, 0.7, 0.9, 0.4, 0.6],
        "network_key": ["gene", "gene", "gene", "gene", "gene"],
    })
    receptor_grn = pd.DataFrame({
        "source": ["REC1", "REC2"],
        "target": ["TF1", "TF2"],
        "score": [1.0, 0.8],
    })
    receptor_graph = pd.DataFrame({
        "source": ["REC1", "REC2"],
        "target": ["REC2", "REC1"],
        "weight": [0.3, 0.2],
    })
    ccc = pd.DataFrame({
        "source": ["LIG1", "LIG2"],
        "target": ["REC1", "REC2"],
        "celltype_source": ["Sender", "Receiver"],
        "celltype_target": ["Receiver", "Sender"],
        "lr_means": [0.9, 0.5],
        "network_key": ["cell_communication", "cell_communication"],
    })
    return grn, receptor_grn, receptor_graph, ccc


def _sort_result(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["multiplex", "node", "layer"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    sort_cols = [col for col in ["multiplex", "layer", "node"] if col in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def _sort_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index().sort_index(axis=1)


def generate_outputs():
    """Run the tiny workflows and return sorted DataFrames for golden tests."""
    grn, receptor_grn, receptor_graph, ccc = tiny_networks()

    celltype = Celltype(
        celltype_name="Sender",
        grn_graph=grn.copy(),
        receptor_grn_bipartite=receptor_grn.copy(),
        receptor_graph=receptor_graph.copy(),
        seeds=["LIG1"],
    )
    celltype_result = _sort_result(
        celltype.explore(restart_proba=0.65, verbose=False)
    )

    sender = Celltype(
        celltype_name="Sender",
        grn_graph=grn.copy(),
        receptor_grn_bipartite=receptor_grn.copy(),
        receptor_graph=receptor_graph.copy(),
    )
    receiver = Celltype(
        celltype_name="Receiver",
        grn_graph=grn.copy(),
        receptor_grn_bipartite=receptor_grn.copy(),
        receptor_graph=receptor_graph.copy(),
    )
    multicell = Multicell(
        celltypes=[sender, receiver],
        cell_communication_graph=ccc.copy(),
        seeds=["LIG1-Sender"],
        verbose=False,
    )
    multicell_result = _sort_result(
        multicell.explore(restart_proba=0.6, verbose=False)
    )

    direct, indirect = multicell_targets(
        seeds=["LIG1", "LIG2"],
        celltypes=["Sender", "Receiver"],
        ccc=ccc.copy(),
        grn=grn.copy(),
        receptor_grn=receptor_grn.copy(),
        receptor_layer=receptor_graph.copy(),
        restart_proba=0.6,
        extend_seeds=True,
        njobs=1,
        verbose=False,
    )

    return {
        "celltype_explore": celltype_result,
        "multicell_explore": multicell_result,
        "multicell_targets_direct": _sort_matrix(direct),
        "multicell_targets_indirect": _sort_matrix(indirect),
    }


def write_outputs(output_dir: Path = GOLDEN_DIR) -> None:
    """Write golden CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = generate_outputs()
    outputs["celltype_explore"].to_csv(output_dir / "celltype_explore.csv", index=False)
    outputs["multicell_explore"].to_csv(output_dir / "multicell_explore.csv", index=False)
    outputs["multicell_targets_direct"].to_csv(output_dir / "multicell_targets_direct.csv")
    outputs["multicell_targets_indirect"].to_csv(output_dir / "multicell_targets_indirect.csv")


if __name__ == "__main__":
    write_outputs()
    print(f"Wrote golden explore outputs to {GOLDEN_DIR}")
