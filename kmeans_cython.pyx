# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
from libc.stdlib cimport malloc, free
import numpy as np
cimport cython
cimport numpy as cnp

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t squared_distance(DTYPE_t* x, DTYPE_t* y, int size) nogil:
    cdef int i
    cdef DTYPE_t dist = 0
    for i in range(size):
        dist += (x[i] - y[i]) ** 2
    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def k_means_cython(cnp.ndarray[DTYPE_t, ndim=2] data, int k, int max_iterations=100):
    cdef int num_samples = data.shape[0]
    cdef int num_features = data.shape[1]
    cdef int i, j, dim, iteration, min_idx
    cdef DTYPE_t min_dist, dist
    cdef int count
    cdef DTYPE_t[:] temp_centroid
    cdef cnp.ndarray[DTYPE_t, ndim=2] centroids = data[np.random.choice(num_samples, k, replace=False), :]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] labels = np.empty(num_samples, dtype=np.int32)
    cdef DTYPE_t[:,:] centroids_view = centroids
    cdef DTYPE_t[:,:] data_view = data

    for iteration in range(max_iterations):
        for i in range(num_samples):
            min_dist = 1e20
            min_idx = -1
            for j in range(k):
                dist = squared_distance(&data_view[i, 0], &centroids_view[j, 0], num_features)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            labels[i] = min_idx
        
        # Recalculate centroids
        for j in range(k):
            count = 0
            temp_centroid = np.zeros(num_features, dtype=DTYPE)
            for i in range(num_samples):
                if labels[i] == j:
                    for dim in range(num_features):
                        temp_centroid[dim] += data_view[i, dim]
                    count += 1
            if count > 0:
                for dim in range(num_features):
                    centroids_view[j, dim] = temp_centroid[dim] / count
                
    return np.asarray(centroids_view), labels
