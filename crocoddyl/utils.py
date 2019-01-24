import numpy as np

'''
Numpy convention.
Let's store vector as 1-d array and matrices as 2-d arrays. Multiplication is done by np.dot.
'''
def raiseIfNan(A,error=None):
    if error is None: error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A))>1e30):
        raise error


def m2a(m): return np.array(m.flat)


def a2m(a): return np.matrix(a).T


def absmax(A): return np.max(abs(A))


def absmin(A): return np.min(abs(A))