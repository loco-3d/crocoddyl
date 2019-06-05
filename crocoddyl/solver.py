import numpy as np


class SolverAbstract:
    """ Abstract class for optimal control solvers.

    In crocoddyl, a solver resolves an optimal control solver which is formulated in a problem abstraction. The main
    routines are computeDirection and tryStep. The former finds a search direction and typically computes the
    derivatives of each action model. The latter rollout the dynamics and cost (i.e. the action) to try the search
    direction found by computeDirection. Both functions used the current guess defined by setCandidate. Finally solve
    function is used to define when the search direction and length are computed in each iterate. It also describes
    the globalization strategy (i.e. regularization) of the numerical optimization.
    """

    def __init__(self, problem):
        # Setting up the problem and allocating the required solver data
        self.problem = problem
        self.allocateData()

        # Allocate common data
        self.xs = [m.State.zero() for m in self.models()]
        self.us = [np.zeros(m.nu) for m in self.problem.runningModels]

        # Default solver parameters
        self.th_acceptStep = 0.1
        self.th_stop = 1e-9
        self.callback = None

    def solve(self, maxiter=100, init_xs=None, init_us=None, isFeasible=False, regInit=None):
        """ Compute the optimal trajectory xopt,uopt as lists of T+1 and T terms.

        From an initial guess init_xs,init_us (feasible or not), iterate over computeDirection and tryStep until
        stoppingCriteria is below threshold. It also describes the globalization strategy used during the numerical
        optimization.
        :param maxiter: maximun allowed number of iterations.
        :param init_xs: initial guess for state trajectory with T+1 elements.
        :param init_us: initial guess for control trajectory with T elements.
        :param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout).
        :param regInit: initial guess for the regularization value. Very low values are typical used with very good
        guess points (init_xs, init_us).
        :returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached.
        """
        raise NotImplementedError("Not implemented yet.")

    def computeDirection(self, recalc=True):
        """ Compute the search direction (dx, du) for the current guess (xs, us).

        You must call setCandidate first in order to define the current guess. A current guess defines a state and
        control trajectory (xs, us) of T+1 and T elements, respectively.
        :params recalc: true for recalculating the derivatives at current state and control.
        :returns the search direction dx, du and the dual lambdas as lists of T+1, T and T+1 lengths.
        """
        raise NotImplementedError("Not implemented yet.")

    def tryStep(self, stepLength):
        """ Try a predefined step length and compute its cost improvement.

        It uses the search direction found by computeDirection to try a determined step length; so you need to run
        first computeDirection. Additionally it returns the cost improvement along the predefined step length.
        :param stepLength: step length
        :returns the cost improvement
        """
        raise NotImplementedError("Not implemented yet.")

    def stoppingCriteria(self):
        """ Return a list of positive values that quantifies the algorithm termination.

        These values typically represents the gradient norm which tell us that it's been reached the local minima. This
        function is used to evaluate the algorithm convergence. The stopping criteria strictly speaking depends on the
        search direction (calculated by computeDirection) but it could also depend on the chosen step length (tested
        by tryStep).
        """
        raise NotImplementedError("Not implemented yet.")

    def expectedImprovement(self):
        """ Return the expected improvement from a given current search direction.

        For computing the expected improvement, you need to compute first the search direction by running
        computeDirection.
        """
        raise NotImplementedError("Not implemented yet.")

    def allocateData(self):
        """ Allocate all the data needed by the solver.
        """
        raise NotImplementedError("Not implemented yet.")

    def setCandidate(self, xs=None, us=None, isFeasible=False):
        """ Set the solver candidate warm-point values (xs, us).

        The solver candidates are defined as a state and control trajectory (xs, us) of T+1 and T elements,
        respectively. Additionally, we need to define is (xs,us) pair is feasible, this means that the dynamics rollout
        give us produces xs.
        :param xs: state trajectory of T+1 elements.
        :param us: control trajectory of T elements.
        :param isFeasible: true if the xs are obtained from integrating the us (rollout).
        """
        if xs is None:
            self.xs[:] = [m.State.zero() for m in self.models()]
        else:
            assert (len(xs) == self.problem.T + 1)
            self.xs[:] = [x.copy() for x in xs]
        if us is None:
            self.us[:] = [np.zeros(m.nu) for m in self.problem.runningModels]
        else:
            assert (len(us) == self.problem.T)
            self.us[:] = [u.copy() for u in us]
        self.isFeasible = isFeasible

    def models(self):
        """ Return a list of all action models."""
        return self.problem.runningModels + [self.problem.terminalModel]

    def datas(self):
        """ Return a list of all action datas."""
        return self.problem.runningDatas + [self.problem.terminalData]

    def setCallbacks(self, callbacks):
        """ Set a list of callback functions using for diagnostic.

        Each iteration, the solver calls these set of functions in order to allowed user the diagnostic of the solver's
        performance.
        :param callbacks: set of callback functions.
        """
        self.callbacks = callbacks
