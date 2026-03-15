#include <stdlib.h>
#include <math.h>
#include "clustering.h"

/* ============================================================================
   Lloyd's K-Means Clustering Implementation (C Extension)
   
   High-performance C implementation for Python extension module.
   Implements Lloyd's algorithm with early convergence detection.
   ============================================================================ */

/**
 * euclidean_distance - Calculate Euclidean distance between two points
 * 
 * Computes: sqrt(sum((p1[i] - p2[i])^2))
 */
static double euclidean_distance(double *p1, double *p2, int dimension) {
    double sum_of_squares = 0.0;
    int i;
    
    for (i = 0; i < dimension; i++) {
        double delta = p1[i] - p2[i];
        sum_of_squares += delta * delta;
    }
    
    return sqrt(sum_of_squares);
}

/**
 * lloyd_clustering - Main clustering algorithm
 * 
 * Algorithm:
 * 1. Initialize centroids from provided initial centroids
 * 2. For each iteration:
 *    a. Assign each point to nearest centroid
 *    b. Update centroids as mean of assigned points
 *    c. Check convergence criterion
 * 3. Return final centroids
 */
double** lloyd_clustering(
    int num_points,
    int dimension,
    int num_clusters,
    int max_iterations,
    double epsilon,
    double** data_points,
    double** initial_centroids)
{
    int i, j, k, iteration;
    double max_centroid_shift;
    
    /* Allocate centroid arrays */
    double **centroids = malloc(num_clusters * sizeof(double*));
    double **new_centroids = malloc(num_clusters * sizeof(double*));
    int *cluster_sizes = malloc(num_clusters * sizeof(int));
    
    for (i = 0; i < num_clusters; i++) {
        centroids[i] = malloc(dimension * sizeof(double));
        new_centroids[i] = malloc(dimension * sizeof(double));
    }
    
    /* Copy initial centroids */
    for (i = 0; i < num_clusters; i++) {
        for (j = 0; j < dimension; j++) {
            centroids[i][j] = initial_centroids[i][j];
        }
    }
    
    /* Main clustering loop */
    for (iteration = 0; iteration < max_iterations; iteration++) {
        
        /* Reset cluster accumulators */
        for (i = 0; i < num_clusters; i++) {
            cluster_sizes[i] = 0;
            for (j = 0; j < dimension; j++) {
                new_centroids[i][j] = 0.0;
            }
        }
        
        /* Assignment step: assign each point to nearest centroid */
        for (i = 0; i < num_points; i++) {
            double min_distance = euclidean_distance(
                data_points[i],
                centroids[0],
                dimension
            );
            int closest_cluster = 0;
            
            for (j = 1; j < num_clusters; j++) {
                double distance = euclidean_distance(
                    data_points[i],
                    centroids[j],
                    dimension
                );
                
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_cluster = j;
                }
            }
            
            /* Accumulate point coordinates */
            cluster_sizes[closest_cluster]++;
            for (j = 0; j < dimension; j++) {
                new_centroids[closest_cluster][j] += data_points[i][j];
            }
        }
        
        /* Update step: recompute centroids */
        for (i = 0; i < num_clusters; i++) {
            if (cluster_sizes[i] > 0) {
                for (j = 0; j < dimension; j++) {
                    new_centroids[i][j] /= cluster_sizes[i];
                }
            } else {
                /* Empty cluster: keep old centroid */
                for (j = 0; j < dimension; j++) {
                    new_centroids[i][j] = centroids[i][j];
                }
            }
        }
        
        /* Convergence check: compute max centroid movement */
        max_centroid_shift = 0.0;
        for (i = 0; i < num_clusters; i++) {
            double shift = euclidean_distance(
                centroids[i],
                new_centroids[i],
                dimension
            );
            
            if (shift > max_centroid_shift) {
                max_centroid_shift = shift;
            }
        }
        
        /* Update centroids for next iteration */
        for (i = 0; i < num_clusters; i++) {
            for (j = 0; j < dimension; j++) {
                centroids[i][j] = new_centroids[i][j];
            }
        }
        
        /* Early termination if converged */
        if (max_centroid_shift < epsilon) {
            break;
        }
    }
    
    /* Free temporary memory */
    for (i = 0; i < num_clusters; i++) {
        free(new_centroids[i]);
    }
    free(new_centroids);
    free(cluster_sizes);
    
    /* Return centroids (caller responsible for freeing) */
    return centroids;
}
