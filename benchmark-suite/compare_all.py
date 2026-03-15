"""
Unified comparison script for all clustering algorithms
"""

import sys
import numpy as np
import time
from pathlib import Path
import json

class ClusteringComparison:
    def __init__(self, n_samples=1000, n_features=50, n_clusters=5):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.data = None
        self.results = {}
        
    def generate_data(self, seed=42):
        """Generate test data"""
        np.random.seed(seed)
        centroids = np.random.randn(self.n_clusters, self.n_features) * 5
        data = []
        for _ in range(self.n_samples):
            center_idx = np.random.randint(0, self.n_clusters)
            point = centroids[center_idx] + np.random.randn(self.n_features)
            data.append(point)
        self.data = np.array(data)
        
    def print_comparison(self):
        """Print formatted comparison"""
        print("\n" + "="*70)
        print(f"CLUSTERING ALGORITHMS COMPARISON")
        print(f"Dataset: {self.n_samples} points, {self.n_features} features")
        print("="*70)
        
        print(f"\n{'Algorithm':<20} {'Time (ms)':<15} {'Quality':<15}")
        print("-"*70)
        
        for algo, metrics in self.results.items():
            print(f"{algo:<20} {metrics.get('time', 0):<15.2f} {metrics.get('quality', 0):<15.4f}")

if __name__ == "__main__":
    comp = ClusteringComparison(n_samples=1000, n_features=50, n_clusters=5)
    comp.generate_data()
    comp.print_comparison()
