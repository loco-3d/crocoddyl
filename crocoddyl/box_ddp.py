import numpy as np
import scipy.linalg as scl

from .ddp import SolverDDP
from .qpsolvers import quadprogWrapper
from .utils import raiseIfNan


class SolverBoxDDP(SolverDDP):
    """ Runs the Control-Limited DDP solver proposed in
    Tassa, Mansard and Todorov, 'Control-Limited Differential Dynamic Programming', ICRA 2014

    This solver uses quadprogWrapper as the default to solve the QP.
    """

    def solve(self,
              maxiter=100,
              init_xs=None,
              init_us=None,
              isFeasible=False,
              regInit=None,
              qpsolver=None,
              ul=None,
              uu=None):
        self.qpsolver = qpsolver if qpsolver is not None else quadprogWrapper
        self.ul = ul
        self.uu = uu
        return SolverDDP.solve(self, maxiter, init_xs, init_us, isFeasible, regInit)

    def computeGains(self, t):
        uu = self.uu
        ul = self.ul
        u = self.us[t]
        nu = self.problem.runningModels[t].nu
        try:
            if ul is None and uu is None:
                qp_d = None
                qp_C = None
            elif uu is not None and ul is None:
                qp_d = uu - u
                qp_C = np.identity(nu)
            elif uu is None and ul is not None:
                qp_d = u - ul
                qp_C = -np.identity(nu)
                # qp_d is the opposite sign, because in the implementation of ddp, k and K are
                # the opposite signs.
            else:
                qp_d = np.hstack([uu - u, u - ul])
                qp_C = np.vstack([-np.identity(nu), np.identity(nu)])
            if self.Quu[t].shape[0] > 0:
                # We have a -Qu[t] here because in the implementation of ddp,
                # k and K are the negative gains
                sol = self.qpsolver(H=self.Quu[t], g=-self.Qu[t], C=qp_C, d=qp_d)
                self.k[t][:] = sol.argmin
                free_sel = ~sol.active
                Quu_f = self.Quu[t][free_sel, :][:, free_sel]
                if sum(free_sel) > 0:
                    Lb = scl.cho_factor(Quu_f)
                    self.K[t][free_sel, :] = scl.cho_solve(Lb, self.Qux[t][free_sel, :])
                    self.K[t][~free_sel, :] = 0.
                else:
                    self.K[t][:, :] = 0.

            else:
                pass
        except scl.LinAlgError:
            raise ArithmeticError('backward error')

    def forwardPass(self, stepLength, warning='ignore'):
        """ Run the forward-pass of the DDP algorithm.

        The forward-pass basically applies a new policy and then rollout the
        system. After this rollouts, it's checked if this policy provides a
        reasonable improvement. For that we use Armijo condition to evaluated the
        chosen step length.
        :param stepLength: step length
        """
        # Argument b is introduce for debug purpose.
        # Argument warning is also introduce for debug: by default, it masks the numpy warnings
        #    that can be reactivated during debug.
        xs, us = self.xs, self.us
        xtry = [self.problem.initialState] + [np.nan] * self.problem.T
        utry = [np.nan] * self.problem.T
        ctry = 0
        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            utry[t] = us[t] - self.k[t] * stepLength - np.dot(self.K[t], m.State.diff(xs[t], xtry[t]))

            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                xnext, cost = m.calc(d, xtry[t], utry[t])
            xtry[t + 1] = xnext.copy()  # not sure copy helpful here.
            ctry += cost
            raiseIfNan([ctry, cost], ArithmeticError('forward error'))
            raiseIfNan(xtry[t + 1], ArithmeticError('forward error'))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            ctry += self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])[1]
        raiseIfNan(ctry, ArithmeticError('forward error'))
        self.xs_try = xtry
        self.us_try = utry
        self.cost_try = ctry
        return xtry, utry, ctry
