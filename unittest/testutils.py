import numpy as np

import pinocchio
from crocoddyl.utils import EPS
from pinocchio.utils import zero


def df_dx(func, x, h=np.sqrt(2 * EPS)):
    """ Perform df/dx by num_diff.
  :params func: function to differentiate f : np.matrix -> np.matrix
  :params x: value at which f is differentiated. type np.matrix
  :params h: eps

  :returns df/dx
  """
    dx = zero(x.size)
    f0 = func(x)
    res = zero([len(f0), x.size])
    for ix in range(x.size):
        dx[ix] = h
        res[:, ix] = (func(x + dx) - f0) / h
        dx[ix] = 0
    return res


def df_dq(model, func, q, h=np.sqrt(2 * EPS)):
    """ Perform df/dq by num_diff. q is in the lie manifold.
    :params func: function to differentiate f : np.matrix -> np.matrix
    :params q: configuration value at which f is differentiated. type np.matrix
    :params h: eps

    :returns df/dq
    """
    dq = zero(model.nv)
    f0 = func(q)
    res = zero([len(f0), model.nv])
    for iq in range(model.nv):
        dq[iq] = h
        res[:, iq] = (func(pinocchio.integrate(model, q, dq)) - f0) / h
        dq[iq] = 0
    return res
