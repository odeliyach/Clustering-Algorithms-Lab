"""
Unit tests for the benchmark comparison framework.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark-suite"))

from compare_all import ClusteringComparison


class TestClusteringComparison:
    def test_generate_data_shape(self):
        comp = ClusteringComparison(n_samples=50, n_features=3, n_clusters=2)
        comp.generate_data(seed=42)
        assert comp.data.shape == (50, 3)

    def test_generate_data_reproducible(self):
        comp1 = ClusteringComparison(n_samples=20, n_features=5, n_clusters=3)
        comp1.generate_data(seed=99)
        comp2 = ClusteringComparison(n_samples=20, n_features=5, n_clusters=3)
        comp2.generate_data(seed=99)
        np.testing.assert_array_equal(comp1.data, comp2.data)

    def test_print_comparison_runs(self, capsys):
        comp = ClusteringComparison()
        comp.generate_data()
        comp.print_comparison()
        captured = capsys.readouterr()
        assert "CLUSTERING ALGORITHMS COMPARISON" in captured.out
