cimport cython
cimport numpy as np
import numpy as np

np.import_array()

DTYPE = np.int64
DTYPE_F = np.float64
ctypedef np.int64_t DTYPE_t
ctypedef np.float64_t DTYPE_F_t


@cython.wraparound(False)
@cython.boundscheck(False)
def strength_new(np.ndarray[DTYPE_F_t, ndim=1] activations,
                 np.ndarray[DTYPE_F_t, ndim=1] resting,
                 list list_of_arrays,
                 list list_of_matrices,
                 DTYPE_F_t minimum,
                 DTYPE_F_t decay,
                 DTYPE_F_t step_size):
    """Fast function for calculating association strength."""
    cdef int i, j, x
    cdef int n_neurons = activations.shape[0]
    cdef int n_arrays = len(list_of_arrays)
    cdef np.ndarray[DTYPE_F_t, ndim=1] net = np.zeros([n_neurons],
                                                      dtype=DTYPE_F)

    cdef np.uintp_t data = np.array((n_arrays,),dtype=np.uintp)
    cdef np.uintp_t shape = np.array((n_arrays,),dtype=np.uintp)

    cdef np.ndarray[double, ndim=1, mode="c"] temp
    cdef np.ndarray[double, ndim=2, mode="c"] mtr

    for i in range(n_arrays):
        temp = list_of_arrays[i]
        mtr = list_of_matrices[i]
        for j in range(temp.shape[0]):
            if temp[j] > 0:
                for x in range(n_neurons):
                    net[x] += temp[j] * mtr[j, x]

    for i in range(n_neurons):
        if net[i] > 0:
            net[i] *= 1.0 - activations[i]
        else:
            net[i] *= activations[i] - minimum
        net[i] -= decay * (activations[i] - resting[i])

    return net * step_size
