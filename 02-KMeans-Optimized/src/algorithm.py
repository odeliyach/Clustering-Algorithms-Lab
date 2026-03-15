"""
K-Means++ Clustering Algorithm

Hybrid implementation combining Python-based initialization with C-accelerated clustering.
"""

import sys
import numpy as np
import pandas as pd
import clustering_engine  # C extension module


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def select_initial_centroids(data, k, random_seed=None):
    """
    K-Means++ Initialization Algorithm
    
    Selects K initial centroids using weighted probability distribution
    to maximize initial spread and reduce convergence time.
    
    Args:
        data: (N, D) array of data points
        k: Number of clusters
        random_seed: Random seed for reproducibility
        
    Returns:
        (K, D) array of initial centroids
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = data.shape[0]
    centroid_indices = []
    
    # Choose first centroid uniformly at random
    first_idx = np.random.choice(n_samples)
    centroid_indices.append(first_idx)
    
    # Choose remaining K-1 centroids
    while len(centroid_indices) < k:
        # Calculate distance from each point to nearest centroid
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):
            if i not in centroid_indices:
                min_dist = min(euclidean_distance(data[i], data[j]) 
                             for j in centroid_indices)
                distances[i] = min_dist
        
        # Normalize distances to probabilities
        probabilities = distances / np.sum(distances)
        
        # Select next centroid with probability proportional to D(x)^2
        new_idx = np.random.choice(n_samples, p=probabilities)
        
        if new_idx not in centroid_indices:
            centroid_indices.append(new_idx)
    
    return np.array([data[i] for i in centroid_indices])


def kmeans_plus_plus_clustering(data, k, max_iter=300, epsilon=0.001, random_seed=None):
    """
    Hybrid K-Means++ Clustering
    
    Phase 1 (Python): K-Means++ initialization
    Phase 2 (C): Lloyd's clustering iterations
    
    Args:
        data: (N, D) array of data points
        k: Number of clusters
        max_iter: Maximum iterations
        epsilon: Convergence threshold
        random_seed: Random seed
        
    Returns:
        centroids: (K, D) final centroid positions
        labels: (N,) cluster assignment for each point
    """
    data = np.asarray(data, dtype=np.float64)
    
    # Phase 1: K-Means++ Initialization (Python)
    initial_centroids = select_initial_centroids(data, k, random_seed)
    
    # Phase 2: Clustering (C Extension)
    final_centroids = clustering_engine.fit(
        data.tolist(),
        initial_centroids.tolist(),
        k,
        max_iter,
        epsilon
    )
    
    # Compute final labels
    final_centroids_array = np.array(final_centroids)
    distances = np.linalg.norm(data[:, None, :] - final_centroids_array[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    
    return final_centroids_array, labels


def load_data_from_csv(filepath):
    """
    Load data from CSV file with ID column.
    
    Format: id,feature1,feature2,...,featureN
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        ids: Array of IDs
        features: (N, D) array of features
    """
    df = pd.read_csv(filepath, header=None)
    ids = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    return ids, features


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python3 algorithm.py <K> [max_iter] [epsilon] <input.csv>")
        sys.exit(1)
    
    # Parse arguments
    try:
        k = int(float(sys.argv[1]))
    except (ValueError, IndexError):
        print("Invalid number of clusters!")
        sys.exit(1)
    
    if k <= 1:
        print("Invalid number of clusters!")
        sys.exit(1)
    
    # Optional parameters
    max_iter = 300
    epsilon = 0.001
    input_file = None
    
    if len(sys.argv) == 3:
        input_file = sys.argv[2]
    elif len(sys.argv) == 4:
        try:
            max_iter = int(float(sys.argv[2]))
        except ValueError:
            print("Invalid maximum iteration!")
            sys.exit(1)
        input_file = sys.argv[3]
    elif len(sys.argv) == 5:
        try:
            max_iter = int(float(sys.argv[2]))
            epsilon = float(sys.argv[3])
        except ValueError:
            print("Invalid parameters!")
            sys.exit(1)
        input_file = sys.argv[4]
    else:
        print("Usage: python3 algorithm.py <K> [max_iter] [epsilon] <input.csv>")
        sys.exit(1)
    
    if max_iter <= 1 or max_iter >= 1000:
        print("Invalid maximum iteration!")
        sys.exit(1)
    
    if epsilon < 0:
        print("Invalid epsilon!")
        sys.exit(1)
    
    # Load data
    ids, features = load_data_from_csv(input_file)
    
    if k >= len(features):
        print("Invalid number of clusters!")
        sys.exit(1)
    
    # Run clustering
    centroids, labels = kmeans_plus_plus_clustering(
        features, k, max_iter, epsilon, random_seed=42
    )
    
    # Output results
    for centroid in centroids:
        print(",".join(f"{x:.4f}" for x in centroid))


if __name__ == "__main__":
    main()
