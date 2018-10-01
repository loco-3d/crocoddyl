import numpy as np
from cddp.data import RunningDDPData, TerminalDDPData
from cddp.utils import isPositiveDefinitive
import time


class DDP(object):
  def __init__(self, system, cost_manager, timeline):
    self.system = system
    self.cost_manager = cost_manager
    self.timeline = timeline
    self.N = len(timeline) - 1

    # Allocation of data for all the DDP intervals and its solution
    self.intervals = []
    self.X_opt = []
    self.U_opt = []
    self.Uff_opt = []
    self.Ufb_opt = []
    for k in range(self.N + 1):
      # Creating the system dynamic and cost data
      sdata = self.system.createData()
      if k == self.N:
        cdata = self.cost_manager.createTerminalData(sdata.ndx)
        self.intervals.append(TerminalDDPData(sdata, cdata))
      else:
        cdata = self.cost_manager.createRunningData(sdata.ndx, sdata.m)
        self.intervals.append(RunningDDPData(sdata, cdata))

        # Creating the solution data
        self.X_opt.append(np.matrix(np.zeros((sdata.nx, 1))))
        self.U_opt.append(np.matrix(np.zeros((sdata.m, 1))))
        self.Uff_opt.append(np.matrix(np.zeros((sdata.m, 1))))
        self.Ufb_opt.append(np.matrix(np.zeros((sdata.m, sdata.ndx))))

    # Setting the time values of the running intervals
    for k in range(self.N):
      it = self.intervals[k]
      it.t0 = timeline[k]
      it.tf = timeline[k+1]

    # Defining the initial and terminal intervals
    self.terminal_interval = self.intervals[-1]
    self.initial_interval = self.intervals[0]

    # Global variables for the DDP algorithm
    self.z = 0.
    self.V_exp = np.matrix(np.zeros(1)) # Backward-pass Value function at t0
    self.V = np.matrix(np.zeros(1)) # Forward-pass Value function at t0
    self.dV_exp = np.matrix(np.zeros(1)) # Expected total cost reduction given alpha
    self.dV = np.matrix(np.zeros(1)) # Total cost reduction
    self.gamma = np.matrix(np.zeros(1)) # sqrt(||Qu^0|| + ... + ||Qu^N||) / N
    self.theta = np.matrix(np.zeros(1)) # sum_{i=0}^N 0.5 * Qu^i * Quu_inv^i * Qu
    self.L = np.matrix(np.zeros((sdata.m, sdata.m))) # Cholesky decomposition of Quu
    self.L_inv = np.matrix(np.zeros((sdata.m, sdata.m))) # Inverse of L
    self.Quu_inv_minus = np.matrix(np.zeros((sdata.m, sdata.m))) # Inverse of -Quu
    self.jt_Quu_j = np.matrix(np.zeros(1))
    self.jt_Qu = np.matrix(np.zeros(1))
    self.muI = np.eye(self.system.m)
    self.n_iter = 0
    self._convergence = False
    self.eps = np.finfo(float).eps
    self.sqrt_eps = np.sqrt(self.eps)

    # Setting up default solver properties
    self.setProperties()

  def setProperties(self):
    """ Sets the properties of the DDP solver.
    """
    # Convergence tolerance and maximum number of iterations
    self.tol = 1e-8
    self.max_iter = 20

    # Regularization parameters (factor and increased rate)
    self.muV = 0.
    self.muLM = 0.
    self.mu0V = 0.
    self.mu0LM = 1e-8
    self.muV_inc = 10.
    self.muV_dec = 0.5
    self.muLM_inc = 10.
    self.muLM_dec = 0.5

    # Line search parameters (step, lower and upper bound of iteration
    # acceptance and decreased rate)
    self.alpha = 1.
    self.alpha_inc = 2.
    self.alpha_dec = 0.5
    self.alpha_min = 1e-3
    self.change_lb = 0.
    self.change_ub = 100.

    # Global variables for analysing solver performance
    self.J_itr = [0.] * self.max_iter
    self.gamma_itr = [0.] * self.max_iter
    self.theta_itr = [0.] * self.max_iter
    self.alpha_itr = [0.] * self.max_iter

  def compute(self, x0, U=None):
    """ Computes the DDP algorithm.
    """
    # Starting time
    start = time.time()

    # Resetting convergence flag
    self._convergence = False

    # Running an initial forward simulation given the initial state and
    # the control sequence
    self.forwardSimulation(x0, U)

    # Resetting mu and alpha. As general rule of thumb we assume that the 
    # quadratic model has some error respect to the true model. So we start
    # closer to steeppest descent by adjusting the initial Levenberg-Marquardt
    # parameter (i.e. mu)
    self.muLM = self.mu0LM
    self.muV = self.mu0V
    self.alpha = 1.

    self.n_iter = 0
    for i in range(self.max_iter):
      # Recording the number of iterations
      self.n_iter = i
      print ("Iteration", self.n_iter, "muV", self.muV,
             "muLM", self.muLM, "alpha", self.alpha)

      # Running the backward sweep
      while not self.backwardPass(self.muLM, self.muV, self.alpha):
        # Quu is not positive-definitive, so increasing the
        # regularization factor
        if self.muLM == 0.:
          self.muLM += self.mu0LM
        else:
          self.muLM *= self.muLM_inc
        print "\t", ("Quu isn't positive. Increasing muLM to", self.muLM)
      # Running the forward pass
      while not self.forwardPass(self.alpha):
        self.alpha *= self.alpha_dec
        print "\t", ("Rejected changes. Decreasing alpha to", self.alpha)
        print "\t", "\t", "Reduction Ratio:", self.z
        print "\t", "\t", "Expected Reduction:", -np.asscalar(self.dV_exp)
        print "\t", "\t", "Actual Reduction:", -np.asscalar(self.dV)
        if self.alpha < self.alpha_min:
          print "\t", ('It cannot be improved solution')
          self._convergence = True
          break

      # Recording the total cost, gradient, theta and alpha for each iteration.
      # This is useful for analysing the solver performance
      self.J_itr[i] = np.asscalar(self.V)
      self.gamma_itr[i] = np.asscalar(self.gamma)
      self.theta_itr[i] = np.asscalar(self.theta)
      self.alpha_itr[i] = self.alpha
      # The quadratic model is accepted so for faster convergence it's better
      # to approach to Newton search direction. We can do it by decreasing the
      # Levenberg-Marquardt parameter
      self.muLM *= self.muLM_dec
      if self.muLM < self.eps: # this is full Newton direction
        self.muLM = self.eps

      # Increasing the stepsize for the next iteration
      self.alpha *= self.alpha_inc
      if self.alpha > 1.:
        self.alpha = 1.


      # Checking convergence
      if self._convergence:
        # Final time
        end = time.time()
        print ("Reached convergence", np.asscalar(self.gamma), " in", end-start, "sec.")

        # Recording the solution
        self._recordSolution()
        return True

    # Final time
    end = time.time()
    print ("Reached allowed iterations", np.asscalar(self.gamma), " in", end-start, "sec.")

    # Recording the solution
    self._recordSolution()
    return False

  def backwardPass(self, muLM, muV, alpha):
    """ Runs the forward pass of the DDP algorithm.

    :param mu: regularization factor
    :param alpha: scaling factor of open-loop control modification (line-search
    strategy)
    """
    # Setting up the final value function as the terminal cost, so we proceed
    # with the backward sweep
    it = self.terminal_interval
    xf = it.x
    l, it.Vx, it.Vxx = \
      self.cost_manager.computeTerminalTerms(self.system, it.cost, xf)

    # Setting up the initial cost value, and the expected reduction equals zero
    self.V_exp[0] = l.copy()
    self.dV_exp[0] = 0.

    # Running the backward sweep
    self.gamma[0] = 0.
    self.theta[0] = 0.
    np.copyto(self.muI, muLM * np.eye(self.system.m)) # np.diag(np.diag(it.Quu.T * it.Quu))
    for k in range(self.N-1, -1, -1):
      it = self.intervals[k]
      it_next = self.intervals[k+1]
      cost_data = it.cost
      system_data = it.system

      # Getting the state, control and step time of the interval
      x = it.x
      u = it.u
      dt = it.tf - it.t0

      # Computing numerically the cost and its derivatives
      l, lx, lu, lxx, luu, lux = \
        self.cost_manager.computeRunningTerms(self.system, cost_data, x, u, dt)

      # Computing the discrete-time linearized model
      fx, fu = self.system.computeLinearModel(system_data, x, u, dt)

      # Getting the value function values of the next interval (prime interval)
      Vx_p = it_next.Vx
      Vxx_p = it_next.Vxx

      # Updating the Q derivatives. Note that this is Gauss-Newton step because
      # we neglect the Hessian, it's also called iLQR.
      np.copyto(it.Qx, lx + fx.T * Vx_p)
      np.copyto(it.Qu, lu + fu.T * Vx_p)
      np.copyto(it.Qxx, lxx + fx.T * Vxx_p * fx)
      np.copyto(it.Quu, luu + fu.T * Vxx_p * fu)
      np.copyto(it.Qux, lux + fu.T * Vxx_p * fx)

      # We apply a two king of regularization called here as Levenberg-Marquat
      # (LM) and Value (V). It's well know that the LM regularization allows us
      # to change the search direction between Newton and steepest descent by
      # increasing muLM. Note that steepest descent provides us guarantee to
      # decrease our cost function but it's too slow especially in very bad
      # scaled problems. Instead Newton direction moves faster towards the
      # minimum but the Hessian cannot approximate well the problem in very
      # nonlinear problems. On the other hand, the Value function regularization
      # smooths the policy function updates; i.e. it reduces changes in the 
      # policy and penalizes changes in the states instead of controls. In
      # practice it reduces the number of iteration in badly posed problems.
      np.copyto(it.Quu_r, it.Quu + self.muI + muV * fu.T * fu)
      if not isPositiveDefinitive(it.Quu_r, self.L):
        return False
      np.copyto(it.Qux_r, it.Qux + muV * fu.T * fx)

      # Computing the feedback and feedforward terms
      np.copyto(self.L_inv, np.linalg.inv(self.L))
      np.copyto(self.Quu_inv_minus, -1. * self.L_inv.T * self.L_inv)
      np.copyto(it.K, self.Quu_inv_minus * it.Qux_r)
      np.copyto(it.j, self.Quu_inv_minus * it.Qu)

      # Computing the value function derivatives of this interval
      np.copyto(self.jt_Quu_j, 0.5 * it.j.T * it.Quu * it.j)
      np.copyto(self.jt_Qu, it.j.T * it.Qu)
      np.copyto(it.Vx, \
                it.Qx + it.K.T * it.Quu * it.j + it.K.T * it.Qu + it.Qux.T * it.j)
      np.copyto(it.Vxx, \
                it.Qxx + it.K.T * it.Quu * it.K + it.K.T * it.Qux + it.Qux.T * it.K)

      # Symmetric can be lost due to round-off error. This ensures the symmetric
      np.copyto(it.Vxx, 0.5 * (it.Vxx + it.Vxx.T))

      # Updating the local cost and expected reduction. The total values are
      # used to check the changes in the forward pass. This method is explained
      # in Tassa's PhD thesis
      self.V_exp[0] += l
      self.dV_exp[0] += alpha * (alpha * self.jt_Quu_j + self.jt_Qu)

      # Updating the theta and gamma given the actual knot
      self.gamma[0] += it.Qu.T * it.Qu
      self.theta[0] -= self.jt_Quu_j + self.jt_Qu

    # Computing the gradient w.r.t. U={u0, ..., uN}
    self.gamma[0] = np.sqrt(self.gamma[0])
    return True

  def forwardPass(self, alpha):
    """ Runs the forward pass of the DDP algorithm.

    :param alpha: scaling factor of open-loop control modification (line-search
    strategy)
    """
    # Initializing the forward pass with the initial state
    it = self.initial_interval
    it.x_new = it.x.copy()
    self.V[0] = 0.

    # Integrate the system along the new trajectory
    for k in range(self.N):
      # Getting the current DDP interval
      it = self.intervals[k]
      it_next = self.intervals[k+1]

      # Computing the new control command
      np.copyto(it.u_new, it.u + alpha * it.j + it.K *
                self.system.differenceConfiguration(it.x_new, it.x))

      # Integrating the system dynamics and updating the new state value
      dt = it.tf - it.t0
      np.copyto(it_next.x_new,
                self.system.stepForward(it.system, it.x_new, it.u_new, dt))

      # Updating the obtained Value function by numerically integrating the
      # running cost function
      self.V[0] += \
        self.cost_manager.computeRunningCost(self.system, it.cost, it.x_new, it.u_new, dt)

    # Including the terminal cost
    it = self.terminal_interval
    self.V[0] += self.cost_manager.computeTerminalCost(self.system, it.cost, it.x_new)

    # Checking convergence of the previous iteration
    if np.abs(self.gamma[0]) <= self.tol:
      self._convergence = True
      return True

    # Checking the changes
    self.dV[0] = self.V - self.V_exp
    self.z = np.asscalar(self.dV) / np.asscalar(self.dV_exp)
    if self.z > self.change_lb and self.z < self.change_ub:
      # Accepting the new trajectory and control, defining them as nominal ones
      for k in range(self.N):
        it = self.intervals[k]
        np.copyto(it.u, it.u_new)
        np.copyto(it.x, it.x_new)
      it = self.terminal_interval
      np.copyto(it.x, it.x_new)
      return True
    else:
      return False

  def forwardSimulation(self, x0, U=None):
    """ Initial forward simulation for starting the DDP algorithm.

    It integrates the system dynamics given an initial state, and a control
    sequence. This provides the initial nominal state trajectory.
    """
    # Setting the initial state
    self.setInitalState(x0)

    # Setting the initial control sequence
    if U != None:
      self.setInitialControlSequence(U)

    # Initializing the forward pass with the initial state
    it = self.initial_interval
    x0 = it.x
    np.copyto(it.x, x0)
    np.copyto(it.x_new, x0)
    self.V_exp[0] = 0.

    # Integrate the system along the initial control sequences
    for k in range(self.N):
      # Getting the current DDP interval
      it = self.intervals[k]
      it_next = self.intervals[k+1]

      # Integrating the system dynamics and updating the new state value
      dt = it.tf - it.t0
      x_next = \
        self.system.stepForward(it.system, it.x, it.u, dt)
      np.copyto(it_next.x, x_next)
      np.copyto(it_next.x_new, x_next)

      # Updating the expected Value function by numerically integrating the
      # running cost function
      self.V_exp[0] += \
        self.cost_manager.computeRunningCost(self.system, it.cost, it.x, it.u, dt)

    # Including the terminal state and cost
    it = self.terminal_interval
    it.x = self.intervals[self.N-1].x
    self.V_exp[0] += \
      self.cost_manager.computeTerminalCost(self.system, it.cost, it.x)

  def setInitalState(self, x0):
    """ Initializes the actual state of the dynamical system.

    :param x0: initial state vector (n-dimensional vector).
    """
    np.copyto(self.initial_interval.x, x0)

  def setInitialControlSequence(self, U):
    """ Initializes the control sequences.

    :param U: initial control sequence (stack of m-dimensional vector).
    """
    assert len(U) == self.N, "Incomplete control sequence."
    for k in range(self.N):
      it = self.intervals[k]
      np.copyto(it.u, U[k])
      np.copyto(it.u_new, U[k])

  def getStateTrajectory(self):
    return self.X_opt

  def getControlSequence(self):
    return self.U_opt

  def getFeedbackGainSequence(self):
    return self.Ufb_opt

  def getFeedforwardSequence(self):
    return self.Uff_opt

  def getTotalCostSequence(self):
    return np.asarray(self.J_itr[:self.n_iter])

  def getConvergenceSequence(self):
    return np.asarray(self.gamma_itr[:self.n_iter]), \
           np.asarray(self.theta_itr[:self.n_iter]), \
           np.asarray(self.alpha_itr[:self.n_iter])

  def saveToFile(self, filename):
    import pickle
    file = open(filename, 'w')
    data = {'T': self.timeline,
            'X': self.getStateTrajectory(),
            'U': self.getControlSequence(),
            'j': self.getFeedforwardSequence(),
            'K': self.getFeedbackGainSequence(),
            'J': self.getTotalCostSequence(),
            'alpha': self.getConvergenceSequence()[2]}
    pickle.dump(data, file)

  def _recordSolution(self):
    for k in range(self.N):
      self.X_opt[k] = self.intervals[k].x_new
      self.U_opt[k] = self.intervals[k].u_new
      self.Uff_opt[k] = self.intervals[k].j
      self.Ufb_opt[k] = self.intervals[k].K
