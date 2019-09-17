import numpy as np

from .kkt import SolverKKT
from .qpsolvers import quadprogWrapper


class SolverBoxKKT(SolverKKT):
    """ Implements the KKT equivalent of the Control-Limited DDP solver proposed in
    Tassa, Mansard and Todorov, 'Control-Limited Differential Dynamic Programming', ICRA 2014

    This solver uses quadprogWrapper as the default to solve the QP.
    """
    def solve(self, maxiter=100, init_xs=None, init_us=None, isFeasible=False, qpsolver=None, ul=None, uu=None):
        self.qpsolver = qpsolver if qpsolver is not None else quadprogWrapper
        self.ul = ul
        self.uu = uu
        return SolverKKT.solve(self, maxiter, init_xs, init_us, isFeasible)

    def computePrimalDual(self):
        uu = self.uu
        ul = self.ul
        ndx = self.ndx
        nu = self.nu
        if ul is None and uu is None:
            qp_d = None
            qp_C = None
        elif uu is not None and ul is None:
            qp_C = np.hstack([np.zeros((nu, ndx)), np.identity(nu)])
            qp_d = np.concatenate([uu - u for u in self.us])
        elif uu is None and ul is not None:
            qp_C = np.hstack([np.zeros((nu, ndx)), -np.identity(nu)])
            qp_d = np.concatenate([u - ul for u in self.us])
        else:
            qp_C = np.hstack([np.zeros((2 * nu, ndx)), np.vstack([np.identity(nu), -np.identity(nu)])])
            qp_d = np.concatenate([uu - u for u in self.us] + [u - ul for u in self.us])

        qp_A = self.jac
        qp_b = -self.cval
        qp_H = self.hess
        qp_g = self.grad
        sol = self.qpsolver(H=qp_H, g=qp_g, A=qp_A, b=qp_b, C=qp_C, d=qp_d)
        self.primaldual = np.concatenate([sol.argmin, sol.dual])
        self.primal = self.primaldual[:self.ndx + self.nu]
        self.dual = self.primaldual[self.ndx + self.nu:]
