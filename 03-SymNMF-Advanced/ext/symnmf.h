#ifndef SYM_NMF_H
#define SYM_NMF_H

double** compute_similarity_matrix_fromX(double** X, int N, int d);
double** compute_diagonal_degre_matrix(double** A, int N);
double** compute_normalized_similarity_matrix(double** A, double** D, int N);

#endif