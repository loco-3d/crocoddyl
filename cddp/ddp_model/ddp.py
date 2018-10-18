def calc(ddpModel, ddpData):
  """Computes all derivatives and stores them.
  Before calling this function, the forward pass has been already run,
  and the state and control trajectories are present in the ddpData.
  """
  self.dynamicsData.computeAllTerms();
  self.systemData.getAllTerms(self.ddpModel.systemModel, self.dynamicsData);
  self.costData.getAllTerms(self.ddpModel.costModel, self.dynamicsData);

  
def forwardSimulation(ddpModel, ddpData, initValue):
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


def solve(ddpModel, ddpData, solverParams, initValue):
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
    print "\t","\t", "--------------------------------------------------Expected Reduction:", -np.asscalar(self.gamma)
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

    if self.debug != None:
      self._recordSolution()
      self.debug.display(self.timeline, self.intervals[0].x, self.X_opt)

  # Final time
  end = time.time()
  print ("Reached allowed iterations", np.asscalar(self.gamma), " in", end-start, "sec.")

  # Recording the solution
  self._recordSolution()
  return False
