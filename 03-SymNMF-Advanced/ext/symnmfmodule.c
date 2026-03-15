#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include <stdlib.h>

#define BETA 0.5

static PyObject* handle_error() {
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}
 
double** pylist_to_cmatrix(PyObject* list_of_lists, int* N, int* d) {
    *N = PyList_Size(list_of_lists);
    if (*N == 0) return NULL;
    *d = PyList_Size(PyList_GetItem(list_of_lists, 0));
    double** M = malloc(*N * sizeof(double*));
    if (M == NULL) {
    handle_error();
    }
    for (int i = 0; i < *N; i++) {
        PyObject* row = PyList_GetItem(list_of_lists, i);
        M[i] = malloc(*d * sizeof(double));
        if (M[i] == NULL) {
            for (int j = 0; j <= i; j++)
             free(M[j]);
            free(M);
            handle_error();
        }
        for (int j = 0; j < *d; j++)
            M[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
    }
    return M;
}

double frobenius_norm_sq(double** A, double** B, int N, int k) {
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < k; j++) {
            double diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    return sum;
}

double** mat_mult(double** A, double** B, int m, int n, int p) {
    double** C = malloc(m * sizeof(double*)); 
    int i,j;
    if (C== NULL) {
    handle_error();
    }
    for ( i = 0; i < m; i++) {
        C[i] = calloc(p, sizeof(double));
        if (C[i] == NULL) {
            for (j = 0; j <= i; j++)
             free(C[j]);
            free(C);
            handle_error();
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

double** transpose(double** A, int m, int n) {
    double** T = malloc(n * sizeof(double*));
    if (T == NULL) { 
        handle_error();
    }
    for (int i = 0; i < n; i++) {
        T[i] = malloc(m * sizeof(double));
        if (T[i] == NULL) {
            for (int j = 0; j <= i; j++) 
             free(T[j]);
            free(T);
            handle_error();
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

double** symnmf_c(double** H, double** W, int N, int k, int max_iter, double eps) {
    int i,j;
    for (int iter = 0; iter < max_iter; iter++) {
        double** WH = mat_mult(W, H, N, N, k);
        double** Ht = transpose(H, N, k);
        double** HHt = mat_mult(H, Ht, N, k, N);
        double** HHtH = mat_mult(HHt, H, N, N, k);
        double** H_new = malloc(N * sizeof(double*));
        if (H_new == NULL) {handle_error();}
        for ( i = 0; i < N; i++) {
            H_new[i] = malloc(k * sizeof(double));
            if (H_new[i] == NULL) {
            for (j = 0; j <= i; j++) free(H_new[j]);
            free(H_new);
            handle_error();
            } 
        }
        for ( i = 0; i < N; i++) {
            for ( j = 0; j < k; j++) {
                double numerator = WH[i][j];
                double denominator = HHtH[i][j];
                if (denominator == 0) denominator = 1e-10;
                double fraction = numerator / denominator;
                H_new[i][j] = H[i][j] * (1 - BETA + BETA * fraction);
            }
        }
        double diff = frobenius_norm_sq(H_new, H, N, k);
        for ( i = 0; i < N; i++) {
            free(WH[i]); free(HHt[i]); free(HHtH[i]);
        }
        free(WH); free(HHt); free(HHtH);
        for ( i = 0; i < k; i++) { free(Ht[i]);   }
        free(Ht); 
        double** H_to_free = H;
        H = H_new;
        if (iter > 0) {
            for (i = 0; i < N; i++) {free(H_to_free[i]);}
            free(H_to_free);
        }
        if (diff < eps) { break;}
    }
    return H;
}

static PyObject* py_sym(PyObject* self, PyObject* args) {
    int i,j;
    (void)self;
    PyObject* list_of_lists;

    if (!PyArg_ParseTuple(args, "O", &list_of_lists)){
        handle_error();
    } 
    int N, d;
    double** X = pylist_to_cmatrix(list_of_lists, &N, &d);

    if (!X) {
        return NULL;
    }
    double** A = compute_similarity_matrix_fromX(X, N, d);
    PyObject* py_result = PyList_New(N);

    for ( i = 0; i < N; i++) {
        PyObject* py_row = PyList_New(N);
        for ( j = 0; j < N; j++)
            PyList_SetItem(py_row, j, PyFloat_FromDouble(A[i][j]));
        PyList_SetItem(py_result, i, py_row);
    }
    for ( i = 0; i < N; i++) {
        free(X[i]);
        free(A[i]);
    }
    free(X);
    free(A);
    return py_result;
}

static PyObject* py_ddg(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* list_of_lists;
    if (!PyArg_ParseTuple(args, "O", &list_of_lists)) {
        handle_error();
    } 
    int N, d;
    double** X = pylist_to_cmatrix(list_of_lists, &N, &d);
    double** A = compute_similarity_matrix_fromX(X, N, d); 
    double** D = compute_diagonal_degre_matrix(A, N);

    PyObject* py_result = PyList_New(N);
    for (int i = 0; i < N; i++) {
        PyObject* py_row = PyList_New(N);
        for (int j = 0; j < N; j++)
            PyList_SetItem(py_row, j, PyFloat_FromDouble(D[i][j]));
        PyList_SetItem(py_result, i, py_row);
    }

    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(X[i]);
        free(D[i]);
    }
    free(A);
    free(X);
    free(D);

    return py_result;
}

static PyObject* py_norm(PyObject* self, PyObject* args) {
    (void)self;
    int N, d;
    PyObject* list_of_lists;
    if (!PyArg_ParseTuple(args, "O", &list_of_lists)) {
        handle_error();
    } 

    double** X = pylist_to_cmatrix(list_of_lists, &N, &d);
    double** A = compute_similarity_matrix_fromX(X, N, d); 
    double** D = compute_diagonal_degre_matrix(A, N);
    double** W = compute_normalized_similarity_matrix(A, D, N);

    PyObject* py_result = PyList_New(N);
    for (int i = 0; i < N; i++) {
        PyObject* py_row = PyList_New(N);
        for (int j = 0; j < N; j++)
            PyList_SetItem(py_row, j, PyFloat_FromDouble(W[i][j]));
        PyList_SetItem(py_result, i, py_row);
    }

    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(X[i]);
        free(D[i]);
        free(W[i]);
    }
    free(A);
    free(X);
    free(D);
    free(W);
    return py_result;
}

static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* py_W;
    PyObject* py_H;
    int max_iter = 300;
    double eps = 1e-4;
    if (!PyArg_ParseTuple(args, "OO|id", &py_H, &py_W, &max_iter, &eps)) {
        handle_error();
    }
  
    int N = PyList_Size(py_W);
    int k = PyList_Size(PyList_GetItem(py_H, 0));
    int W_cols = N; 

    double** W = pylist_to_cmatrix(py_W, &N, &W_cols); 
    double** H = pylist_to_cmatrix(py_H, &N, &k); 
    double** H_new = symnmf_c(H, W, N, k, max_iter, eps);

    PyObject* py_result = PyList_New(N);
    for (int i = 0; i < N; i++) {
        PyObject* py_row = PyList_New(k);
        for (int j = 0; j < k; j++)
            PyList_SetItem(py_row, j, PyFloat_FromDouble(H_new[i][j]));
        PyList_SetItem(py_result, i, py_row);
    }
    for (int i = 0; i < N; i++) { 
        free(W[i]); free(H_new[i]);free(H[i]);
    } 
    free(W);
    free(H);
    free(H_new);

    return py_result;
}

static PyMethodDef SymnmfMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Perform full SymNMF"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymnmfMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}
