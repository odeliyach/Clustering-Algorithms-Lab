import sys
import numpy as np
from sklearn.metrics import silhouette_score
import math
import symnmfmodule
import pandas as pd

epsilon = 1e-4

def dist(v1 ,v2):
    """
    Compute Euclidean distance between two vectors.
    """
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) ** 2
    return math.sqrt(sum)


def create_clusters(data, centroids):
    """
    Assign each data point to the nearest centroid, forming clusters.
    Returns the clusters and their corresponding labels.
    """
    clusters = [[] for i in range(K)]
    labels = []
    for datapoint in data:
        dists = [dist(datapoint, centroid) for centroid in centroids]
        assigned_cluster = dists.index(min(dists))
        labels.append(assigned_cluster)
        clusters[assigned_cluster].append(datapoint)
    return clusters, labels


def update_centroids(prev_centroids, clusters):
    """
    Calculate new centroids as the mean of points in each cluster.
    """
    new_centroids = []
    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:
            new_centroids.append(prev_centroids[i])
        else:
            k = len(cluster)
            centroid = [sum(coordinate)/k for coordinate in zip(*cluster)]
            new_centroids.append(centroid)
    return new_centroids

def Kmeans(K, data):
    """
    Perform K-means clustering from hw on the dataset.
    Iterates until centroids converge or max iterations reached.
    Returns cluster labels for data points.
    """
    centroids = data[:K]
    for i in range(300):
        clusters, labels = create_clusters(data, centroids)
        new_centroids = update_centroids(centroids, clusters)
        delta_mew = [dist(new_centroids[i], centroids[i]) for i in range(K)]
        if max(delta_mew) < epsilon:
            break
        centroids = new_centroids
    return labels

def SymNMF(K, data):
    """
    Perform SymNMF.
    Initializes factorization matrix and runs the SymNMF algorithm.
    Returns cluster labels based on factorization results.
    """
    np.random.seed(1234)
    N, d = data.shape 
    W = np.array(symnmfmodule.norm(data.tolist()))
    m = np.mean(W)
    initH = np.random.uniform(0, 2 * np.sqrt(m / K), size=(N, K))
    resultH = symnmfmodule.symnmf(initH.tolist(), W.tolist(),300, 1e-4)
    labels = np.argmax(resultH, axis=1)
    return labels
try:
    K = int(sys.argv[1]) 
    filename = sys.argv[2]    
    data = np.loadtxt(filename, delimiter=",")
    if data.ndim == 1:
         data = data.reshape(-1, 1)
    if data.size == 0:
        raise ValueError 
    N, d = data.shape
    if K < 0 or K >= N:
        raise ValueError 
    kmeans_labels = Kmeans(K, data)
    symnmf_labels = SymNMF(K, data)
    score_kmeans = silhouette_score(data, kmeans_labels)
    score_symnmf = silhouette_score(data, symnmf_labels)
    np.savetxt("labels.csv", symnmf_labels, delimiter=",", fmt="%.5f")
    print(f"nmf: {score_symnmf:.4f}")
    print(f"kmeans: {score_kmeans:.4f}")
except ValueError:
    print("An Error Has Occurred")
    sys.exit(1)
except Exception:
    print("An Error Has Occurred")
    sys.exit(1)
