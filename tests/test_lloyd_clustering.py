"""
Unit tests for Lloyd's K-Means Clustering (Phase 1 - Basic).

Tests cover core algorithmic functions: distance computation, point
assignment, centroid recomputation, and end-to-end convergence.
"""

import sys
import os
import math
import pytest

# Allow imports from 01-KMeans-Basic
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01-KMeans-Basic"))

from lloyd_clustering import (
    euclidean_distance,
    assign_points_to_nearest_center,
    recompute_representatives,
    lloyd_clustering_algorithm,
    CONVERGENCE_EPSILON,
)


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------

class TestEuclideanDistance:
    def test_identical_points(self):
        assert euclidean_distance([0, 0], [0, 0]) == 0.0

    def test_unit_distance(self):
        assert euclidean_distance([0, 0], [1, 0]) == pytest.approx(1.0)

    def test_2d_diagonal(self):
        assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_3d_points(self):
        assert euclidean_distance([1, 2, 3], [4, 6, 3]) == pytest.approx(5.0)

    def test_negative_coordinates(self):
        assert euclidean_distance([-1, -1], [2, 3]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# assign_points_to_nearest_center
# ---------------------------------------------------------------------------

class TestAssignPoints:
    def test_simple_assignment(self):
        dataset = [[0, 0], [10, 10]]
        centroids = [[0, 0], [10, 10]]
        clusters = assign_points_to_nearest_center(dataset, centroids, 2)
        assert clusters[0] == [[0, 0]]
        assert clusters[1] == [[10, 10]]

    def test_three_clusters(self):
        dataset = [[0, 0], [1, 0], [10, 10], [11, 10], [20, 0], [21, 0]]
        centroids = [[0, 0], [10, 10], [20, 0]]
        clusters = assign_points_to_nearest_center(dataset, centroids, 3)
        assert len(clusters[0]) == 2
        assert len(clusters[1]) == 2
        assert len(clusters[2]) == 2

    def test_equidistant_chooses_first(self):
        """When equidistant, index of the min selects the first centroid."""
        dataset = [[5, 0]]
        centroids = [[0, 0], [10, 0]]
        clusters = assign_points_to_nearest_center(dataset, centroids, 2)
        assert clusters[0] == [[5, 0]]


# ---------------------------------------------------------------------------
# recompute_representatives
# ---------------------------------------------------------------------------

class TestRecomputeRepresentatives:
    def test_single_point_cluster(self):
        centroids = recompute_representatives(
            [[0, 0]], [[[3, 4]]], 1
        )
        assert centroids == [[3, 4]]

    def test_mean_of_two_points(self):
        centroids = recompute_representatives(
            [[0, 0]], [[[2, 4], [4, 6]]], 1
        )
        assert centroids[0] == pytest.approx([3.0, 5.0])

    def test_empty_cluster_retains_centroid(self):
        prev = [[5, 5]]
        centroids = recompute_representatives(prev, [[]], 1)
        assert centroids == [[5, 5]]


# ---------------------------------------------------------------------------
# lloyd_clustering_algorithm  (end-to-end)
# ---------------------------------------------------------------------------

class TestLloydClustering:
    def test_two_obvious_clusters(self):
        cluster_a = [[0, 0], [1, 0], [0, 1], [1, 1]]
        cluster_b = [[10, 10], [11, 10], [10, 11], [11, 11]]
        dataset = cluster_a + cluster_b

        centroids = lloyd_clustering_algorithm(
            dataset, num_clusters=2, max_iterations=300,
            convergence_epsilon=CONVERGENCE_EPSILON,
        )

        # Centroids should be near (0.5, 0.5) and (10.5, 10.5)
        centroids_sorted = sorted(centroids, key=lambda c: c[0])
        assert centroids_sorted[0][0] == pytest.approx(0.5, abs=0.1)
        assert centroids_sorted[1][0] == pytest.approx(10.5, abs=0.1)

    def test_convergence_within_max_iter(self):
        """Algorithm should converge in fewer than max_iterations for well-separated data."""
        dataset = [[i, 0] for i in range(5)] + [[i, 100] for i in range(5)]
        centroids = lloyd_clustering_algorithm(
            dataset, num_clusters=2, max_iterations=300,
            convergence_epsilon=CONVERGENCE_EPSILON,
        )
        assert len(centroids) == 2

    def test_three_clusters_3d(self):
        """Verify clustering works in 3-D space using the repo's example data."""
        dataset = [
            [1.5, 2.3, 4.1],
            [2.1, 2.8, 3.9],
            [1.8, 2.5, 4.0],
            [5.2, 5.1, 5.3],
            [5.0, 4.9, 5.2],
            [9.1, 9.8, 9.2],
            [9.3, 9.5, 9.1],
        ]
        centroids = lloyd_clustering_algorithm(
            dataset, num_clusters=3, max_iterations=300,
            convergence_epsilon=CONVERGENCE_EPSILON,
        )
        assert len(centroids) == 3
