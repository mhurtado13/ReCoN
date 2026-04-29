"""
Tests for recon.data.load_data module.

Tests the receptor gene loading functionality.
"""

import pytest
import pandas as pd
import importlib
from recon.data.load_data import (
    TUTORIAL_DATA_REGISTRY,
    download_tutorial,
    fetch_all_tutorial_data,
    fetch_tutorial_data,
    load_receptor_genes,
    receptor_gene_resources,
)
import os

load_data_module = importlib.import_module("recon.data.load_data")


class TestReceptorGeneResources:
    """Test the receptor_gene_resources constant."""
    
    def test_resources_list_exists(self):
        """Test that receptor_gene_resources is defined."""
        assert isinstance(receptor_gene_resources, list)
        assert len(receptor_gene_resources) > 0
    
    def test_resources_list_content(self):
        """Test that expected resources are in the list."""
        assert "human_receptor_gene_from_NichenetPKN" in receptor_gene_resources
        assert "mouse_receptor_gene_from_NichenetPKN" in receptor_gene_resources


class TestLoadReceptorGenes:
    """Test load_receptor_genes function."""
    
    def test_load_human_receptor_genes(self):
        """Test loading human receptor genes."""
        df = load_receptor_genes("human_receptor_gene_from_NichenetPKN")
        
        # Should return a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Should have required columns
        assert "source" in df.columns
        assert "target" in df.columns
        
        # Should not be empty
        assert len(df) > 0
        
        # Source should be receptors (genes), target should be downstream genes
        assert all(isinstance(val, str) for val in df["source"].head())
        assert all(isinstance(val, str) for val in df["target"].head())
    
    def test_load_mouse_receptor_genes(self):
        """Test loading mouse receptor genes."""
        df = load_receptor_genes("mouse_receptor_gene_from_NichenetPKN")
        
        # Should return a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Should have required columns
        assert "source" in df.columns
        assert "target" in df.columns
        
        # Should not be empty
        assert len(df) > 0
    
    def test_human_vs_mouse_different(self):
        """Test that human and mouse data are different."""
        human_df = load_receptor_genes("human_receptor_gene_from_NichenetPKN")
        mouse_df = load_receptor_genes("mouse_receptor_gene_from_NichenetPKN")
        
        # Should have different data (different gene names)
        # Not all genes should be identical
        human_genes = set(human_df["source"].unique())
        mouse_genes = set(mouse_df["source"].unique())
        
        # There should be some differences
        assert human_genes != mouse_genes
    
    def test_invalid_resource_name(self):
        """Test that invalid resource names raise ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            load_receptor_genes("invalid_receptor_list")
    
    def test_typo_in_resource_name(self):
        """Test that typos in resource names raise ValueError."""
        with pytest.raises(ValueError):
            load_receptor_genes("human_receptor_gene_from_Nichenet")  # Missing PKN
    
    def test_none_as_input(self):
        """Test that None as input raises appropriate error."""
        with pytest.raises((ValueError, TypeError)):
            load_receptor_genes(None)
    
    def test_empty_string_as_input(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            load_receptor_genes("")
    
    def test_dataframe_structure(self):
        """Test the structure of returned DataFrames."""
        df = load_receptor_genes("human_receptor_gene_from_NichenetPKN")
        
        # Should have at least source and target columns
        assert len(df.columns) >= 2
        
        # No null values in critical columns
        assert df["source"].notna().all()
        assert df["target"].notna().all()
    
    def test_load_multiple_times_same_result(self):
        """Test that loading the same resource multiple times gives consistent results."""
        df1 = load_receptor_genes("human_receptor_gene_from_NichenetPKN")
        df2 = load_receptor_genes("human_receptor_gene_from_NichenetPKN")
        
        # Should be identical - compare shapes and values
        assert df1.shape == df2.shape
        assert list(df1.columns) == list(df2.columns)
        assert df1.equals(df2)
    
    def test_receptor_genes_are_unique_per_target(self):
        """Test that receptor-gene pairs are properly structured."""
        df = load_receptor_genes("mouse_receptor_gene_from_NichenetPKN")
        
        # Each receptor can have multiple target genes
        # Reset index to avoid RangeIndex assertion issues in pandas groupby
        receptor_counts = df.reset_index(drop=True).groupby("source").size()
        
        # Should have multiple targets per receptor on average
        assert receptor_counts.mean() >= 1
        
        # But total pairs should be reasonable
        assert len(df) > len(df["source"].unique())


class TestDataIntegration:
    """Test integration with ReCoN's data structures."""
    
    def test_compatible_with_celltype_construction(self):
        """Test that loaded data is compatible with Celltype construction."""
        df = load_receptor_genes("mouse_receptor_gene_from_NichenetPKN")
        
        # Should be usable as receptor_grn_bipartite
        # Check it has the minimal required columns
        required_cols = ["source", "target"]
        for col in required_cols:
            assert col in df.columns
        
        # Data types should be strings (gene names)
        assert df["source"].dtype == object
        assert df["target"].dtype == object
    
    def test_receptor_names_format(self):
        """Test that receptor names don't have special suffixes yet."""
        df = load_receptor_genes("human_receptor_gene_from_NichenetPKN")
        
        # Source should be plain gene names (no _receptor suffix yet)
        # That suffix is added by Celltype constructor
        sample_receptors = df["source"].head(10)
        
        # Most should not end with _receptor (that's added later)
        non_suffixed = sum(1 for r in sample_receptors if not r.endswith("_receptor"))
        assert non_suffixed >= 5  # Most should be plain names


class TestFetchTutorialData:
    """Test fetch_tutorial_data function with the smallest file."""

    def test_unknown_tutorial_file_raises_value_error(self):
        """Role: reject tutorial filenames that are not in the registry."""
        with pytest.raises(ValueError, match="Unknown file"):
            fetch_tutorial_data("missing/file.h5ad")

    def test_missing_pooch_dependency_raises_helpful_import_error(self, monkeypatch):
        """Role: explain the tutorial extra when pooch is unavailable."""
        monkeypatch.setitem(__import__("sys").modules, "pooch", None)

        with pytest.raises(ImportError, match="recon\\[tutorials\\]"):
            fetch_tutorial_data("perturbation_tuto/rna.h5ad")

    def test_existing_file_short_circuits_download(self, tmp_path, monkeypatch):
        """Role: avoid network work when a cached tutorial file already exists."""
        existing = tmp_path / "data" / "perturbation_tuto" / "rna.h5ad"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"cached")

        def fail_retrieve(*args, **kwargs):
            raise AssertionError("pooch.retrieve should not be called")

        monkeypatch.setattr("pooch.retrieve", fail_retrieve)

        path = fetch_tutorial_data(
            "perturbation_tuto/rna.h5ad",
            data_dir=str(tmp_path / "data"),
            force=False,
        )

        assert path == str(existing)

    def test_force_download_calls_pooch_with_registry_hash(self, tmp_path, monkeypatch):
        """Role: verify Zenodo URL, hash, filename, path, and force behavior."""
        existing = tmp_path / "data" / "perturbation_tuto" / "rna.h5ad"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"stale")
        calls = {}

        def fake_retrieve(**kwargs):
            calls.update(kwargs)
            return "/tmp/downloaded/rna.h5ad"

        monkeypatch.setattr("pooch.retrieve", fake_retrieve)

        path = fetch_tutorial_data(
            "perturbation_tuto/rna.h5ad",
            data_dir=str(tmp_path / "data"),
            force=True,
        )

        assert path == "/tmp/downloaded/rna.h5ad"
        assert calls["url"].endswith("/rna.h5ad")
        assert calls["known_hash"] == TUTORIAL_DATA_REGISTRY["perturbation_tuto/rna.h5ad"]
        assert calls["fname"] == "perturbation_tuto/rna.h5ad"
        assert calls["path"] == str(tmp_path / "data")
        assert calls["progressbar"] is True

    def test_fetch_all_tutorial_data_downloads_each_registry_entry(self, monkeypatch):
        """Role: ensure the bulk downloader iterates over the full tutorial registry."""
        calls = []

        def fake_fetch(filename, data_dir="./data", force=False):
            calls.append((filename, data_dir, force))
            return f"/tmp/{filename}"

        monkeypatch.setattr(load_data_module, "fetch_tutorial_data", fake_fetch)

        paths = fetch_all_tutorial_data(data_dir="/tmp/tutorials", force=True)

        assert set(paths) == set(TUTORIAL_DATA_REGISTRY)
        assert calls == [
            (filename, "/tmp/tutorials", True)
            for filename in TUTORIAL_DATA_REGISTRY
        ]

    def test_fetch_smallest_file(self, tmp_path, monkeypatch):
        """Test downloading the smallest tutorial file."""
        # Define the smallest file to test
        smallest_file = "perturbation_tuto/rna.h5ad"
        
        # Temporary directory for testing
        test_data_dir = tmp_path / "data"
        test_data_dir.mkdir()

        def fake_retrieve(url, known_hash, fname, path, progressbar):
            downloaded = os.path.join(path, fname)
            os.makedirs(os.path.dirname(downloaded), exist_ok=True)
            with open(downloaded, "wb") as handle:
                handle.write(b"mock h5ad")
            return downloaded

        monkeypatch.setattr("pooch.retrieve", fake_retrieve)

        # Fetch the file
        file_path = fetch_tutorial_data(smallest_file, data_dir=str(test_data_dir))

        # Check if the file exists
        assert os.path.exists(file_path)

        # Check if the file is not empty
        assert os.path.getsize(file_path) > 0

    def test_download_tutorial_single_file(self, monkeypatch):
        """Test download_tutorial delegates to single-file download."""
        calls = {}

        def fake_fetch(filename, data_dir="./data", force=False):
            calls["filename"] = filename
            calls["data_dir"] = data_dir
            calls["force"] = force
            return "/tmp/rna.h5ad"

        monkeypatch.setattr(
            "recon.data.load_data.fetch_tutorial_data",
            fake_fetch
        )

        path = download_tutorial(
            "perturbation_tuto/rna.h5ad",
            data_dir="/tmp/data",
            force=True
        )

        assert path == "/tmp/rna.h5ad"
        assert calls == {
            "filename": "perturbation_tuto/rna.h5ad",
            "data_dir": "/tmp/data",
            "force": True,
        }

    def test_download_tutorial_all_files(self, monkeypatch):
        """Test download_tutorial delegates to all-file download."""
        calls = {}

        def fake_fetch_all(data_dir="./data/perturbation_tuto", force=False):
            calls["data_dir"] = data_dir
            calls["force"] = force
            return {"file": "/tmp/file"}

        monkeypatch.setattr(
            "recon.data.load_data.fetch_all_tutorial_data",
            fake_fetch_all
        )

        paths = download_tutorial(data_dir="/tmp/data", force=True)

        assert paths == {"file": "/tmp/file"}
        assert calls == {"data_dir": "/tmp/data", "force": True}
