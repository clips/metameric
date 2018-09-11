cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport fabs, sqrt, exp, cos, pow

DTYPE = np.int64
DTYPE_F = np.float64
ctypedef np.int64_t DTYPE_t
ctypedef np.float64_t DTYPE_F_t


cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

@cython.wraparound(False)
@cython.boundscheck(False)
def strength(np.ndarray[DTYPE_F_t, ndim=1] conn,
             np.ndarray[DTYPE_F_t, ndim=2] mtr):
    """Fast function for calculating association strength."""
    cdef int i, j
    cdef int n_conns = conn.shape[0]
    cdef int n_neurons = mtr.shape[1]
    cdef np.ndarray[DTYPE_F_t, ndim=1] net = np.zeros([n_neurons],
                                                      dtype=DTYPE_F)

    for i in range(n_conns):
        if conn[i] > 0:
            for j in range(n_neurons):
                net[j] += conn[i] * mtr[i, j]

    return net

@cython.wraparound(False)
@cython.boundscheck(False)
def strength_new(np.ndarray[DTYPE_F_t, ndim=1] activations,
             np.ndarray[DTYPE_F_t, ndim=1] resting,
             np.ndarray[DTYPE_F_t, ndim=1] conn,
             np.ndarray[DTYPE_F_t, ndim=2] mtr,
             DTYPE_F_t minimum,
             DTYPE_F_t decay,
             DTYPE_F_t step_size):
    """Fast function for calculating association strength."""
    cdef int i, j
    cdef int n_conns = conn.shape[0]
    cdef int n_neurons = mtr.shape[1]
    cdef np.ndarray[DTYPE_F_t, ndim=1] net = np.zeros([n_neurons],
                                                      dtype=DTYPE_F)

    for i in range(n_conns):
        if conn[i] > 0:
            for j in range(n_neurons):
                net[j] += conn[i] * mtr[i, j]

    for i in range(n_neurons):
        if net[i] > 0:
            net[i] *= 1.0 - activations[i]
        else:
            net[i] *= activations[i] - minimum
        net[i] -= decay * (activations[i] - resting[i])

    return net * step_size


@cython.wraparound(False)
@cython.boundscheck(False)
def strength_grossberg(np.ndarray[DTYPE_F_t, ndim=1] activations,
             np.ndarray[DTYPE_F_t, ndim=1] resting,
             np.ndarray[DTYPE_F_t, ndim=1] conn,
             np.ndarray[DTYPE_F_t, ndim=2] mtr,
             DTYPE_F_t minimum,
             DTYPE_F_t decay):

    cdef int i, j
    cdef int n_conns = conn.shape[0]
    cdef int n_neurons = mtr.shape[1]
    cdef np.ndarray[DTYPE_F_t, ndim=1] exc = np.zeros([n_neurons],
                                                      dtype=DTYPE_F)
    cdef np.ndarray[DTYPE_F_t, ndim=1] inh = np.zeros([n_neurons],
                                                      dtype=DTYPE_F)


    for i in range(n_conns):
        if conn[i] > 0:
            for j in range(n_neurons):
                if mtr[i, j] > 0:
                    exc[j] += conn[i] * mtr[i, j]
                else:
                    inh[j] += conn[i] * mtr[i, j]

    for i in range(n_neurons):
        exc[i] *= (1.0 - activations[i])
        exc[i] -= (activations[i] - minimum) * inh[i]
        exc[i] -= decay * (activations[i] - resting[i])

    return exc
