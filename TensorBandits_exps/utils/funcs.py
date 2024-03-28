import numpy as np


def L_inf_norm(tensor):
    return np.sum(np.abs(tensor))

def prod(arg):
    """ returns the product of elements in arg.
    arg can be list, tuple, set, and array with numerical values. """
    ret = 1
    for i in range(0, len(arg)):
        ret = ret * arg[i]
    return ret


def marginal_multiplication(X, Y, axis):
    result = np.tensordot(X, Y, axes=([axis], [1]))
    result = np.moveaxis(result, len(result.shape) - 1, axis)
    return result