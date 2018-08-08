import numpy as np
from cddp.data import RunningDDPData, TerminalDDPData
from cddp.utils import isPositiveDefinitive


class DDP(object):
  def __init__(self, dynamics, cost_manager, timeline):
    self.dynamics = dynamics
    self.cost_manager = cost_manager
    self.timeline = timeline
    self.N = len(timeline) - 1

    # Allocation of data for all the DDP intervals
    self.intervals = []
    for k in range(self.N + 1):
      # Creating the dynamic and cost data
      ddata = self.dynamics.createData()
      if k == self.N:
        cdata = self.cost_manager.createTerminalData(ddata.nv)
        self.intervals.append(TerminalDDPData(ddata, cdata))
      else:
        cdata = self.cost_manager.createRunningData(ddata.nv, ddata.m)
        self.intervals.append(RunningDDPData(ddata, cdata))
    
    # Global variables for the DDP algorithm
    self.V = np.matrix(np.zeros(1))
    self.V_new = np.matrix(np.zeros(1))
    self.dV = np.matrix(np.zeros(1))
    self.dV_exp = np.matrix(np.zeros(1))

    # Setting the time values of the running intervals
    for k in range(self.N):
      it = self.intervals[k]
      it.t0 = timeline[k]
      it.tf = timeline[k+1]

    # Defining the inital and terminal intervals
    self.terminal_interval = self.intervals[-1]
    self.initial_interval = self.intervals[0]

    # Convergence tolerance and maximum number of iterations
    self.tol = 1e-3
    self.max_iter = 1

    # Regularization parameters (factor and increased rate)
    self.mu = 0.
    self.increased_rate = 10.

    # Line search parameters (step, lower and upper bound of iteration
    # acceptance and decreased rate)
    self.alpha = 1.
    self.change_lb = 0.
    self.change_ub = 100.
    self.decreased_rate = 2.

  def compute(self, x0, U=None):
    """ Computes the DDP algorithm.
    """
    # Running an initial forward simulation given the initial state and
    # the control sequence
    self.forwardSimulation(x0, U)

    # Resetting mu and alpha
    self.mu = 0.
    self.alpha = 1.

    self.n_iter = 0
    for i in range(self.max_iter):
      # Recording the number of iterations
      self.n_iter = i

      # Running the backward sweep
      while not self.backwardPass(self.mu, self.alpha):
        # Quu is not positive-definitive, so increasing the
        # regularization factor
        if self.mu == 0.:
          self.mu += 1e-8
        else:
          self.mu *= self.increased_rate

      # Running the forward pass
      while not self.forwardPass(self.alpha):
        self.alpha /= self.decreased_rate
        if self.alpha == 0.:
          print 'No found solution'
          break
        if self.alpha < 1e-8:
          self.alpha = 0.

      # Checking convergence
      if abs(self.dV) <= self.tol:
        return True

    return False

  def backwardPass(self, mu, alpha):
    """ Runs the forward pass of the DDP algorithm.

    :param mu: regularization factor
    :param alpha: scaling factor of open-loop control modification (line-search
    strategy)
    """
    # Setting up the final value function as the terminal cost, so we proceed
    # with the backward sweep
    it = self.terminal_interval
    xf = it.x
    l, it.Vx, it.Vxx = self.cost_manager.computeTerminalTerms(it.cost, xf)

    # Setting up the initial cost value, and the expected reduction equals zero
    self.V[0] = l.copy()
    self.dV_exp[0] = 0.

    # Running the backward sweep
    for k in range(self.N-1, -1, -1):
      it = self.intervals[k]
      it_next = self.intervals[k+1]
      cost_data = it.cost
      dyn_data = it.dynamics

      # Getting the state, control and step time of the interval
      x = it.x
      u = it.u
      dt = it.tf - it.t0

      # Computing the cost and its derivatives
      l, lx, lu, lxx, luu, lux = \
        self.cost_manager.computeRunningTerms(cost_data, x, u)

      # Integrating the derivatives of the cost function
      # TODO we need to use the integrator class for this
      l *= dt
      lx *= dt
      lu *= dt
      lxx *= dt
      luu *= dt
      lux *= dt

      # Computing the discrete time dynamics derivatives
      fx, fu = self.dynamics.computeDerivatives(dyn_data, x, u, dt)

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

      # We apply a regularization on the Quu and Qux as Tassa.
      # This regularization is needed when the Hessian is not
      # positive-definitive or when the minimum is far and the
      # quadratic model is inaccurate
      np.copyto(it.Quu_r, it.Quu + mu * fu.T * fu)
      if not isPositiveDefinitive(it.Quu_r):
        return False
      np.copyto(it.Qux_r, it.Qux + mu * fu.T * fx)

      # Computing the feedback and feedforward terms
      L_inv = np.linalg.inv(np.linalg.cholesky(it.Quu_r))
      Quu_inv = L_inv.T * L_inv
      np.copyto(it.K, - Quu_inv * it.Qux_r)
      np.copyto(it.j, - Quu_inv * it.Qu)

      # Computing the value function derivatives of this interval
      jt_Quu_j = 0.5 * it.j.T * it.Quu * it.j
      jt_Qu = it.j.T * it.Qu
      np.copyto(it.Vx, \
                it.Qx + it.K.T * it.Quu * it.j + it.K.T * it.Qu + it.Qux.T * it.j)
      np.copyto(it.Vxx, \
                it.Qxx + it.K.T * it.Quu * it.K + it.K.T * it.Qux + it.Qux.T * it.K)
      
      # Symmetric can be lost due to round-off error. This ensures the symmetric
      np.copyto(it.Vxx, 0.5 * (it.Vxx + it.Vxx.T))
      
      # Updating the local cost and expected reduction. The total values are
      # used to check the changes in the forward pass. This is method is
      # explained in Tassa's PhD thesis
      self.V[0] += l
      self.dV_exp[0] += alpha * (alpha * jt_Quu_j + jt_Qu)
    return True

  def forwardPass(self, alpha):
    """ Runs the forward pass of the DDP algorithm.

    :param alpha: scaling factor of open-loop control modification (line-search
    strategy)
    """
    # Initializing the forward pass with the initial state
    it = self.initial_interval
    it.x_new = it.x.copy()
    self.V_new[0] = 0.

    # Integrate the system along the new trajectory
    for k in range(self.N):
      # Getting the current DDP interval
      it = self.intervals[k]
      it_next = self.intervals[k+1]

      # Computing the new control command
      np.copyto(it.u_new, it.u + alpha * it.j + it.K *
                self.dynamics.stateDifference(it.x_new, it.x))

      # Integrating the dynamics and updating the new state value
      dt = it.tf - it.t0
      np.copyto(it_next.x_new,
                self.dynamics.stepForward(it.dynamics, it.x_new, it.u_new, dt))

      # Integrating the cost and updating the new value function
      # TODO we need to use the integrator class for this
      self.V_new[0] += \
        self.cost_manager.computeRunningCost(it.cost, it.x_new, it.u_new) * dt

    # Including the terminal cost
    it = self.terminal_interval
    self.V_new[0] += self.cost_manager.computeTerminalCost(it.cost, it.x_new)

    # Checking the changes
    self.dV[0] = self.V_new - self.V
    z = self.dV[0, 0] / self.dV_exp[0, 0]
    print "Expected Reduction:", -self.dV_exp[0, 0]
    print "Actual Reduction:", -self.dV[0, 0]
    print "Reduction Ratio", z
    if z > self.change_lb and z < self.change_ub:
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

    It integrates the system's dynamics given an initial state, and a control
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
    self.V[0] = 0.

    # Integrate the system along the initial control sequences
    for k in range(self.N):
      # Getting the current DDP interval
      it = self.intervals[k]
      it_next = self.intervals[k+1]

      # Integrating the dynamics and updating the new state value
      dt = it.tf - it.t0
      x_next = \
        self.dynamics.stepForward(it.dynamics, it.x, it.u, dt)
      np.copyto(it_next.x, x_next)
      np.copyto(it_next.x_new, x_next)

      # Integrating the cost and updating the new value function
      self.V[0] += \
        self.cost_manager.computeRunningCost(it.cost, it.x, it.u) * dt

    # Including the terminal state and cost
    it = self.terminal_interval
    it.x = self.intervals[self.N-1].x
    self.V[0] += self.cost_manager.computeTerminalCost(it.cost, it.x)

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

  # def getControlTrajectory(self):

  #   control_traj = np.matrix(np.zeros((self.model.m,self.N)))
  #   for k,it in enumerate(self.intervals[:-1]):
  #     control_traj[:,k] = it.u

  #   return control_traj

  # def getStateTrajectory(self):

  #   state_traj = np.matrix(np.zeros((self.model.n,self.N+1)))
  #   for k,it in enumerate(self.intervals[:-1]):
  #     state_traj[:,k] = it.x0

  #   state_traj[:,-1] = self.terminal_interval.x

  #   return state_traj
