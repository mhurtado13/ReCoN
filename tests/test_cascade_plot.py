"""Tests for recon.plot cascade plots."""
import importlib.util
from pathlib import Path
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


_CASCADE_CORE_PATH = Path(__file__).parents[1] / "src" / "recon" / "plot" / "cascade_core.py"
_SPEC = importlib.util.spec_from_file_location("cascade_core", _CASCADE_CORE_PATH)
cascade_core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cascade_core)


class TestCascadeFlow:
    """Test cascade plot flow direction handling."""

    def test_draw_single_edge_upstream_direction(self, monkeypatch):
        captured = []

        def mock_add_patch(patch):
            if isinstance(patch, FancyArrowPatch):
                captured.append(patch._posA_posB)

        _, ax = plt.subplots()
        monkeypatch.setattr(ax, "add_patch", mock_add_patch)

        cascade_core.draw_single_edge(
            ax=ax,
            src="LIGAND::Sender",
            tgt="RECEPTOR::Receiver",
            value=1.0,
            layer_key="lig_rec",
            rad=0.1,
            positions={
                "LIGAND::Sender": (0.0, 0.0),
                "RECEPTOR::Receiver": (1.0, 0.0),
            },
            lw_ranges={"lig_rec": (0.0, 1.0)},
            color_fn=lambda _node, _layer: (0.5, 0.5, 0.5, 0.5),
            flow="upstream",
        )

        assert captured[0] == [(0.0, 0.0), (1.0, 0.0)]
        plt.close("all")

    def test_draw_single_edge_downstream_direction(self, monkeypatch):
        captured = []

        def mock_add_patch(patch):
            if isinstance(patch, FancyArrowPatch):
                captured.append(patch._posA_posB)

        _, ax = plt.subplots()
        monkeypatch.setattr(ax, "add_patch", mock_add_patch)

        cascade_core.draw_single_edge(
            ax=ax,
            src="LIGAND::Sender",
            tgt="RECEPTOR::Receiver",
            value=1.0,
            layer_key="lig_rec",
            rad=0.1,
            positions={
                "LIGAND::Sender": (0.0, 0.0),
                "RECEPTOR::Receiver": (1.0, 0.0),
            },
            lw_ranges={"lig_rec": (0.0, 1.0)},
            color_fn=lambda _node, _layer: (0.5, 0.5, 0.5, 0.5),
            flow="downstream",
        )

        assert captured[0] == [(1.0, 0.0), (0.0, 0.0)]
        plt.close("all")

    def test_invalid_flow_raises(self):
        with pytest.raises(ValueError, match="flow must be either"):
            cascade_core.normalize_flow("sideways")
