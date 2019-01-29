import time
import numpy as np
from crocoddyL.utils import isPositiveDefinitive, EPS


class DDPSolver(object):
  @staticmethod
  def solve(ddpModel, ddpData, ddpParams):
    """ Run the DDP solver.

    The solver computes a optimal trajectory and control commmands by iteratives
    running backward and forward passes. The backward-pass updates locally the
    quadratic approximation of the problem, and the forward-pass rollout this
    new policy by integrating the system dynamics along a tuple of control
    commands U.
    :param ddpModel: DDP model
    :param ddpData: entired DDP data
    :param ddpParams: DDP solver parameters
    """
    start = time.time()

    # Resetting convergence flag
    ddpData._convergence = False

    # Running an initial forward simulation. calculates the forward dynamics and
    # total costs
    DDPSolver.forwardSimulation(ddpModel, ddpData)

    ddpData.muLM = ddpParams.mu0LM
    ddpData.muV = ddpParams.mu0V
    ddpData.alpha = ddpParams.alpha0

    ddpData.n_iter = 0
    for i in xrange(ddpParams.max_iter):
      # Resetting flags
      ddpData.backward_status = " "
      ddpData.forward_status = " "

      # Recording the number of iterations
      ddpData.n_iter = i

      # Update the quadratic approximation
      DDPSolver.updateQuadraticAppr(ddpModel, ddpData)

      while not DDPSolver.backwardPass(ddpModel, ddpData, ddpParams):
        # Quu is not positive-definitive, so increasing the
        # regularization factor
        if ddpData.muLM == 0.:
          ddpData.muLM = ddpParams.mu0LM
        else:
          ddpData.muLM *= ddpParams.muLM_inc
        ddpData.backward_status = "r" # Regularization of the search direction

      # Running the forward pass
      while not DDPSolver.forwardPass(ddpModel, ddpData, ddpParams):
        ddpData.alpha *= ddpParams.alpha_dec
        ddpData.forward_status = "s" # Reduction of the step-length
        if ddpData.alpha < ddpParams.alpha_min:
          print "\t", ('It cannot be improved solution')
          ddpData._convergence = True
          break

      # Printing message
      if ddpParams.print_level:
        if i % 10 == 0:
          print "iter \t cost \t       theta \t   gamma \t muV \t     muLM \talpha"
        print "%4i  %0.5e  %0.5e  %0.5e  %10.5e  %0.5e%c  %0.4f%c" % \
          (ddpData.n_iter, ddpData.totalCost,
          ddpData.theta, ddpData.gamma,
          ddpData.muV, ddpData.muLM, ddpData.backward_status,
          ddpData.alpha, ddpData.forward_status)

      # Recording the total cost, gradient, weighted gradient (gamme),
      # regularization values and alpha for each iteration. This is useful for
      # analysing the solver performance
      if ddpParams.record:
        ddpParams.cost_itr[i] = ddpData.totalCost
        ddpParams.gamma_itr[i] = ddpData.gamma
        ddpParams.theta_itr[i] = ddpData.theta
        ddpParams.muLM_itr[i] = ddpData.muLM
        ddpParams.muV_itr[i] = ddpData.muV
        ddpParams.alpha_itr[i] = ddpData.alpha

      # The quadratic model is accepted so for faster convergence it's better
      # to approach to Newton search direction. We can do it by decreasing the
      # Levenberg-Marquardt parameter
      ddpData.muLM *= ddpParams.muLM_dec

      if ddpData.muLM < EPS: # this is full Newton direction
        ddpData.muLM = EPS

      # This regularization smooth the policy updates. Experimentally it helps
      # to reduce the number of iteration whenever the problem isn't well posed
      ddpData.muV *= ddpParams.muV_dec
      if ddpData.muV < EPS:
        ddpData.muV = EPS

      # Increasing the stepsize for the next iteration
      ddpData.alpha *= ddpParams.alpha_inc
      if ddpData.alpha > 1.:
        ddpData.alpha = 1.

      # Checking convergence
      if ddpData._convergence:
        if ddpParams.print_level:
          # Final time
          end = time.time()
          print
          print "EXIT: Optimal Solution Found in %0.4f sec." % (end-start)

        # Recording the solution
        #TODO: Log solution
        """
        self._recordSolution()
        """
        return True

      #TODO: Log solution
      """
      if self.debug != None:
        self._recordSolution()
        self.debug.display(self.timeline, self.intervals[0].x, self.X_opt)
      """
    # Final time
    end = time.time()
    print ("Reached allowed iterations", ddpData.gamma, " in", end-start, "sec.")

    # Recording the solution
    #TODO
    #self._recordSolution()
    return False

  @staticmethod
  def forwardPass(ddpModel, ddpData, ddpParams):
    """ Run the forward-pass of the DDP algorithm.

    The forward-pass basically applies a new policy and then rollout the
    system. After this rollouts, it's checked if this policy provides a
    reasonable improvement. For that we use Armijo condition to evaluated the
    choosen step length.
    :param ddpModel: DDP model
    :param ddpData: entired DDP data
    :param ddpParams: DDP solver parameters
    """
    # Integrate the system along the new trajectory and compute its cost
    ddpData.totalCost = 0.
    for k in xrange(ddpData.N):
      # Getting the current DDP interval
      it = ddpData.interval[k]

      # Computing the new control command
      np.copyto(it.u, it.u_prev +\
        ddpData.alpha * it.j + \
        np.dot(it.K, ddpModel.dynamicModel.differenceState(it.dynamic,
                                                           it.x_prev,
                                                           it.x)))

      # Integrating the system dynamics and updating the new state value
      ddpModel.forwardRunningCalc(
        ddpData.interval[k].dynamic,
        ddpData.interval[k].cost,
        ddpData.interval[k].x,
        ddpData.interval[k].u,
        ddpData.interval[k+1].x)
      ddpData.totalCost += ddpData.interval[k].cost.l
    ddpModel.forwardTerminalCalc(
      ddpData.interval[-1].dynamic,
      ddpData.interval[-1].cost,
      ddpData.interval[-1].x)
    ddpData.totalCost += ddpData.interval[-1].cost.l

    # Checking convergence of the current iteration
    if np.abs(ddpData.gamma) <= ddpParams.tol:
      ddpData._convergence = True
      return True

    # Updating the actual and expected Value functions
    ddpData.dV_exp = \
      ddpData.alpha * (ddpData.alpha * ddpData.total_jt_Quu_j + ddpData.total_jt_Qu)
    ddpData.dV = ddpData.totalCost - ddpData.totalCost_prev

    # Checking the changes
    ddpData.z_new = ddpData.dV / ddpData.dV_exp
    if ddpData.z_new > ddpParams.armijo_condition \
       and ddpData.z_new < ddpParams.change_ub:
      ddpData.z = ddpData.z_new
      return True
    else:
      return False

  @staticmethod
  def backwardPass(ddpModel, ddpData, ddpParams):
    """ Run the backward-pass of the DDP algorithm.

    The backward-pass is equivalent to a Riccati recursion. It updates the
    quadratic terms of the optimal control problem, and the gradient and
    Hessian of the value function. Additionally, it computes the new
    feedforward and feedback commands (i.e. control policy). A regularization
    scheme is used to ensure a good search direction. The norm of the gradient,
    a the directional derivatives are computed.
    :param ddpModel: DDP model
    :param ddpData: entired DDP data
    :param ddpParams: DDP solver parameters
    """
    # Setting up the initial values
    ddpData.total_jt_Quu_j = 0.
    ddpData.total_jt_Qu = 0.

    # Running the backward sweep
    ddpData.gamma = 0.
    ddpData.theta = 0.
    np.copyto(ddpData.muI,
              ddpData.muLM * np.identity(ddpModel.dynamicModel.nu()))
    np.copyto(ddpData.interval[-1].Vx,
              ddpData.interval[-1].cost.lx)
    np.copyto(ddpData.interval[-1].Vxx,
              ddpData.interval[-1].cost.lxx)
    for k in xrange(ddpData.N-1, -1, -1):
      it = ddpData.interval[k]

      # Getting the value function values of the next interval (prime interval)
      fx = it.dynamic.discretizer.fx
      fu = it.dynamic.discretizer.fu

      # Getting the Value function derivatives of the next interval
      Vx = ddpData.interval[k+1].Vx
      Vxx = ddpData.interval[k+1].Vxx

      # Updating the Q derivatives. Note that this is Gauss-Newton step because
      # we neglect the Hessian, it's also called iLQR.
      np.copyto(it.Vpxx_fu, np.dot(Vxx, fu))
      np.copyto(it.Quu, it.cost.luu + np.dot(fu.T, it.Vpxx_fu))
      np.copyto(it.Quu_r, it.Quu + ddpData.muI + ddpData.muV * np.dot(fu.T, fu))
      if not isPositiveDefinitive(it.Quu_r, it.L):
        return False
      np.copyto(it.Vpxx_fx, np.dot(Vxx, fx))
      np.copyto(it.Qx, it.cost.lx + np.dot(fx.T, Vx))
      np.copyto(it.Qu, it.cost.lu + np.dot(fu.T, Vx))
      np.copyto(it.Qxx, it.cost.lxx + np.dot(fx.T, it.Vpxx_fx))
      np.copyto(it.Qux, it.cost.lux + np.dot(fu.T, it.Vpxx_fx))
      np.copyto(it.Qux_r, it.Qux + ddpData.muV * np.dot(fu.T, fx))

      # Computing the feedback and feedforward terms
      np.copyto(it.L_inv, np.linalg.inv(it.L))
      np.copyto(it.Quu_inv_minus, -1. * np.dot(it.L_inv.T, it.L_inv))
      np.copyto(it.K, np.dot(it.Quu_inv_minus, it.Qux_r))
      np.copyto(it.j, np.dot(it.Quu_inv_minus, it.Qu))

      # Computing the Value function derivatives of this interval
      it.jt_Quu_j = 0.5 * np.asscalar(np.dot(it.j.T, np.dot(it.Quu_r, it.j)))
      it.jt_Qu = np.asscalar(np.dot(it.j.T, it.Qu))
      np.copyto(it.Vx, \
                it.Qx + np.dot(it.K.T, np.dot(it.Quu_r, it.j)) +\
                np.dot(it.K.T, it.Qu) + np.dot(it.Qux_r.T ,it.j))
      np.copyto(it.Vxx, \
                it.Qxx + np.dot(it.K.T, np.dot(it.Quu_r, it.K)) +\
                np.dot(it.K.T, it.Qux_r) + np.dot(it.Qux_r.T, it.K))

      # Symmetric can be lost due to round-off error. This ensures the symmetric
      np.copyto(it.Vxx, 0.5 * (it.Vxx + it.Vxx.T))

      # Updating terms for computing the expected reduction in the Value function
      ddpData.total_jt_Quu_j += it.jt_Quu_j
      ddpData.total_jt_Qu += it.jt_Qu

      # Updating the theta and gamma given the actual knot
      ddpData.gamma += np.asscalar(np.dot(it.Qu.T, it.Qu))
      ddpData.theta -= it.jt_Quu_j + it.jt_Qu

    # Computing the norm of the cost gradient w.r.t. U={u0, ..., uN}
    ddpData.gamma = np.sqrt(ddpData.gamma)
    ddpData.totalCost_prev = ddpData.totalCost
    return True

  @staticmethod
  def forwardSimulation(ddpModel, ddpData):
    """ Integrate the dynamics and compute its cost given an initial condition

    The initial condition is defined by the initial state and initial sequence
    of control commands. It represents the warm-point of the optimal control
    problem.
    :param ddpModel: DDP model
    :param ddpData: entired DDP data
    """
    # Integrate the system along the new trajectory and compute its cost
    ddpData.totalCost = 0.
    for k in xrange(ddpData.N):
      # Integrating the system dynamics and updating the new state value
      ddpModel.forwardRunningCalc(
        ddpData.interval[k].dynamic,
        ddpData.interval[k].cost,
        ddpData.interval[k].x,
        ddpData.interval[k].u,
        ddpData.interval[k+1].x)
      ddpData.totalCost += ddpData.interval[k].cost.l
    ddpModel.forwardTerminalCalc(
      ddpData.interval[-1].dynamic,
      ddpData.interval[-1].cost,
      ddpData.interval[-1].x)
    ddpData.totalCost += ddpData.interval[-1].cost.l
    return

  @staticmethod
  def updateQuadraticAppr(ddpModel, ddpData):
    """ Update the quadratic approximation of the problem.
    
    :param ddpModel: entired DDP model
    :param ddpData: entired DDP data
    """
    # Updating along the horizon. TODO: Parallelize.
    for k in xrange(ddpData.N):
      # Copying the current state and control into the previous ones
      np.copyto(ddpData.interval[k].x_prev,
                ddpData.interval[k].x)
      np.copyto(ddpData.interval[k].u_prev,
                ddpData.interval[k].u)
      ddpModel.backwardRunningCalc(
        ddpData.interval[k].dynamic,
        ddpData.interval[k].cost,
        ddpData.interval[k].x,
        ddpData.interval[k].u)
    # Copying the current state into the previous one
    np.copyto(ddpData.interval[-1].x_prev,
              ddpData.interval[-1].x)
    ddpModel.backwardTerminalCalc(
      ddpData.interval[-1].dynamic,
      ddpData.interval[-1].cost,
      ddpData.interval[-1].x)

  @staticmethod
  def getStateTrajectory(ddpData):
    """ Return the state trajectory contained in the ddpData.

    :param ddpData: entired DDP data
    """
    X = []
    for i in xrange(ddpData.N):
      X.append(ddpData.interval[i].x)
    X.append(ddpData.interval[-1].x)
    return X

  @staticmethod
  def getControlSequence(ddpData):
    """ Return the control sequence contained in the ddpData.

    :param ddpData: entired DDP data
    """
    U = []
    for i in xrange(ddpData.N):
      U.append(ddpData.interval[i].u)
    return U

  @staticmethod
  def getFeedbackGainSequence(ddpData):
    """ Return the feedback gain sequences contained in the ddpData.

    :param ddpData: entired DDP data
    """
    K = []
    for i in xrange(ddpData.N):
      K.append(ddpData.interval[i].K)
    return K

  @staticmethod
  def getFeedforwardSequence(ddpData):
    """ Return the feedforward control sequence contained in the ddpData.

    :param ddpData: entired DDP data
    """
    j = []
    for i in xrange(ddpData.N):
      j.append(ddpData.interval[i].j)
    return j
