"""Golden-output regression tests for tiny explore workflows."""
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden"
GENERATOR_PATH = ROOT / "scripts" / "generate_golden_explore_outputs.py"
RTOL = 1e-12
ATOL = 1e-15


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_golden_explore_outputs",
        GENERATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_golden_outputs():
    direct = pd.read_csv(
        GOLDEN_DIR / "multicell_targets_direct.csv",
        index_col=0,
    )
    direct.columns.name = "celltype_target"
    return {
        "celltype_explore": pd.read_csv(GOLDEN_DIR / "celltype_explore.csv"),
        "multicell_explore": pd.read_csv(GOLDEN_DIR / "multicell_explore.csv"),
        "multicell_targets_direct": direct,
        "multicell_targets_indirect": pd.read_csv(
            GOLDEN_DIR / "multicell_targets_indirect.csv",
            header=[0, 1],
            index_col=0,
        ),
    }


def test_tiny_explore_workflows_match_golden_outputs():
    """Role: detect end-to-end numerical regressions in tiny explore workflows."""
    generator = _load_generator()
    actual = generator.generate_outputs()
    expected = _read_golden_outputs()

    for key in ("celltype_explore", "multicell_explore"):
        pd.testing.assert_frame_equal(
            actual[key],
            expected[key],
            check_dtype=False,
            check_exact=False,
            rtol=RTOL,
            atol=ATOL,
        )

    for key in ("multicell_targets_direct", "multicell_targets_indirect"):
        pd.testing.assert_frame_equal(
            actual[key],
            expected[key],
            check_dtype=False,
            check_exact=False,
            rtol=RTOL,
            atol=ATOL,
        )
