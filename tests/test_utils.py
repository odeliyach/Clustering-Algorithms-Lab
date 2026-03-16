"""
Unit tests for clustering utility functions (Phase 2 - Optimized).

Tests cover data validation, parameter validation, inertia computation,
and normalization / denormalization.
"""

import sys
import os
import numpy as np
import pytest

# Allow imports from 02-KMeans-Optimized/src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02-KMeans-Optimized", "src"))

from utils import (
    validate_data,
    validate_parameters,
    compute_inertia,
    normalize_data,
    denormalize_centroids,
)


# ---------------------------------------------------------------------------
# validate_data
# ---------------------------------------------------------------------------

class TestValidateData:
    def test_valid_2d_array(self):
        result = validate_data([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            validate_data([1, 2, 3])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_data(np.empty((0, 2)))

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_data([[1, float("nan")]])

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="infinite"):
            validate_data([[1, float("inf")]])


# ---------------------------------------------------------------------------
# validate_parameters
# ---------------------------------------------------------------------------

class TestValidateParameters:
    def test_valid_params(self):
        validate_parameters(k=3, max_iter=300, epsilon=0.001, n_samples=100)

    def test_k_too_small(self):
        with pytest.raises(ValueError):
            validate_parameters(k=1, max_iter=300, epsilon=0.001, n_samples=100)

    def test_k_exceeds_n(self):
        with pytest.raises(ValueError):
            validate_parameters(k=100, max_iter=300, epsilon=0.001, n_samples=100)

    def test_max_iter_out_of_range(self):
        with pytest.raises(ValueError):
            validate_parameters(k=3, max_iter=1000, epsilon=0.001, n_samples=100)

    def test_negative_epsilon(self):
        with pytest.raises(ValueError):
            validate_parameters(k=3, max_iter=300, epsilon=-1.0, n_samples=100)


# ---------------------------------------------------------------------------
# compute_inertia
# ---------------------------------------------------------------------------

class TestComputeInertia:
    def test_zero_inertia(self):
        data = np.array([[0, 0], [1, 1]])
        centroids = np.array([[0, 0], [1, 1]])
        labels = np.array([0, 1])
        assert compute_inertia(data, centroids, labels) == pytest.approx(0.0)

    def test_positive_inertia(self):
        data = np.array([[0, 0], [2, 0]])
        centroids = np.array([[1, 0]])
        labels = np.array([0, 0])
        # Each point is distance 1 from centroid → inertia = 1 + 1 = 2
        assert compute_inertia(data, centroids, labels) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# normalize_data / denormalize_centroids
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_round_trip(self):
        data = np.array([[2, 4], [4, 8], [6, 12]], dtype=np.float64)
        normed, mean, std = normalize_data(data)
        recovered = denormalize_centroids(normed, mean, std)
        np.testing.assert_allclose(recovered, data)

    def test_zero_variance_column(self):
        data = np.array([[1, 5], [1, 10]], dtype=np.float64)
        normed, mean, std = normalize_data(data)
        # Constant column should stay at 0 after normalization
        assert normed[0, 0] == pytest.approx(0.0)
        assert normed[1, 0] == pytest.approx(0.0)
