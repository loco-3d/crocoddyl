import numpy as np

EPS = np.finfo(float).eps

'''
Numpy convention.
Let's store vector as 1-d array and matrices as 2-d arrays. Multiplication is done by np.dot.
'''
def raiseIfNan(A,error=None):
    if error is None: error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A))>1e30):
        raise error


def m2a(m): return np.array(m).squeeze()


def a2m(a): return np.matrix(a).T


def absmax(A): return np.max(abs(A))


def absmin(A): return np.min(abs(A))

def randomOrthonormalMatrix(dim=3):
     """ Create a random orthonormal matrix using np.random.rand
     """
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H
