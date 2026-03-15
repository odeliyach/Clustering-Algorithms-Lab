#!/usr/bin/env python3
"""
Lloyd's K-Means Clustering Implementation - Basic Version

This module implements Lloyd's k-means clustering algorithm in pure Python.
Partitions N datapoints in d dimensions into K clusters using iterative
assignment and centroid updates with Euclidean distance metric.

Mathematical Background:
- Euclidean distance: d(p,q) = sqrt(sum((p_i - q_i)^2))
- Centroid update: μ_k = (1/|C_k|) * sum(x_i for x_i in C_k)
- Convergence: max(d(μ_k^t, μ_k^(t+1))) < epsilon
"""

import sys
import math

# Algorithm constants
DEFAULT_MAX_ITERATIONS = 400
CONVERGENCE_EPSILON = 0.001


def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: List of coordinates
        point2: List of coordinates
        
    Returns:
        Euclidean distance as float
    """
    sum_of_squares = 0
    for i in range(len(point1)):
        delta = point1[i] - point2[i]
        sum_of_squares += delta ** 2
    return math.sqrt(sum_of_squares)


def assign_points_to_nearest_center(dataset, cluster_representatives, num_clusters):
    """
    Assignment step: Assign each datapoint to its nearest centroid.
    
    Args:
        dataset: List of datapoints (each is a list of coordinates)
        cluster_representatives: List of current centroids
        num_clusters: Number of clusters
        
    Returns:
        List of clusters, where each cluster is a list of assigned datapoints
    """
    clusters = [[] for _ in range(num_clusters)]
    
    for datapoint in dataset:
        distances = [euclidean_distance(datapoint, centroid)
                    for centroid in cluster_representatives]
        assigned_cluster = distances.index(min(distances))
        clusters[assigned_cluster].append(datapoint)
    
    return clusters


def recompute_representatives(previous_representatives, clusters, num_clusters):
    """
    Update step: Recompute centroids as mean of assigned points.
    
    Args:
        previous_representatives: List of previous centroids
        clusters: List of clusters from assignment step
        num_clusters: Number of clusters
        
    Returns:
        List of new centroids
    """
    new_representatives = []
    
    for i in range(num_clusters):
        cluster = clusters[i]
        
        if len(cluster) == 0:
            # Empty cluster: keep previous centroid
            new_representatives.append(previous_representatives[i])
        else:
            # Compute mean of all points in cluster
            point_count = len(cluster)
            centroid = [sum(coordinate) / point_count
                       for coordinate in zip(*cluster)]
            new_representatives.append(centroid)
    
    return new_representatives


def parse_command_line_arguments():
    """
    Parse and validate command-line arguments.
    
    Returns:
        Tuple: (num_clusters, max_iterations)
    """
    if len(sys.argv) <= 1:
        print("Incorrect number of clusters!")
        sys.exit(1)
    
    elif len(sys.argv) == 2:
        try:
            num = float(sys.argv[1])
        except ValueError:
            print("Incorrect number of clusters!")
            sys.exit(1)
        
        if not num.is_integer() or num <= 1:
            print("Incorrect number of clusters!")
            sys.exit(1)
        
        num_clusters = int(num)
        max_iterations = DEFAULT_MAX_ITERATIONS
    
    elif len(sys.argv) == 3:
        try:
            num_clusters_float = float(sys.argv[1])
        except ValueError:
            print("Incorrect number of clusters!")
            sys.exit(1)
        
        try:
            max_iterations_float = float(sys.argv[2])
        except ValueError:
            print("Incorrect maximum iteration!")
            sys.exit(1)
        
        if not num_clusters_float.is_integer() or num_clusters_float <= 1:
            print("Incorrect number of clusters!")
            sys.exit(1)
        
        if (not max_iterations_float.is_integer() or
            max_iterations_float <= 1 or max_iterations_float >= 1000):
            print("Incorrect maximum iteration!")
            sys.exit(1)
        
        num_clusters = int(num_clusters_float)
        max_iterations = int(max_iterations_float)
    
    else:
        print("Incorrect number of clusters!")
        sys.exit(1)
    
    return num_clusters, max_iterations


def read_dataset_from_stdin():
    """
    Read datapoints from stdin in CSV format.
    
    Returns:
        List of datapoints (each datapoint is a list of floats)
    """
    dataset = []
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            datapoint = [float(x) for x in line.split(",")]
            dataset.append(datapoint)
        except ValueError:
            print("Invalid input format!")
            sys.exit(1)
    
    return dataset


def lloyd_clustering_algorithm(dataset, num_clusters, max_iterations, convergence_epsilon):
    """
    Lloyd's K-Means Clustering Algorithm.
    
    Algorithm:
    1. Initialize centroids as first K datapoints
    2. For each iteration:
       a. Assign each point to nearest centroid (assignment step)
       b. Update centroids as mean of assigned points (update step)
       c. Check if max centroid movement < epsilon (convergence)
    3. Return final centroids
    
    Args:
        dataset: List of datapoints
        num_clusters: Number of clusters K
        max_iterations: Maximum number of iterations
        convergence_epsilon: Convergence threshold
        
    Returns:
        List of final centroids
    """
    # Initialize centroids as first K datapoints
    cluster_representatives = dataset[:num_clusters]
    
    # Main iteration loop
    for iteration_count in range(max_iterations):
        
        # Assignment step
        clusters = assign_points_to_nearest_center(dataset,
                                                  cluster_representatives,
                                                  num_clusters)
        
        # Update step
        new_representatives = recompute_representatives(cluster_representatives,
                                                       clusters,
                                                       num_clusters)
        
        # Convergence check: compute max centroid movement
        centroid_shifts = [euclidean_distance(new_representatives[i],
                                             cluster_representatives[i])
                          for i in range(num_clusters)]
        max_centroid_shift = max(centroid_shifts)
        
        # Update centroids for next iteration
        cluster_representatives = new_representatives
        
        # Early termination if converged
        if max_centroid_shift < convergence_epsilon:
            break
    
    return cluster_representatives


def main():
    """Main entry point."""
    
    # Parse command-line arguments
    num_clusters, max_iterations = parse_command_line_arguments()
    
    # Read dataset from stdin
    dataset = read_dataset_from_stdin()
    
    # Validate dataset size
    if num_clusters >= len(dataset):
        print("Incorrect number of clusters!")
        sys.exit(1)
    
    # Run Lloyd's clustering algorithm
    final_centroids = lloyd_clustering_algorithm(dataset,
                                                num_clusters,
                                                max_iterations,
                                                CONVERGENCE_EPSILON)
    
    # Output final centroids
    for centroid in final_centroids:
        print(",".join(f"{coordinate:.4f}" for coordinate in centroid))


if __name__ == "__main__":
    main()
