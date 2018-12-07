import time
import numpy as np
from cddp.utils import isPositiveDefinitive, EPS


class Solver(object):
  @staticmethod
  def solve(ddpModel, ddpData, solverParams):
    """ Run the DDP solver.

    The solver computes a optimal trajectory and control commmands by iteratives
    running backward and forward passes. The backward-pass updates locally the
    quadratic approximation of the problem, and the forward-pass rollout this
    new policy by integrating the system dynamics along a tuple of control
    commands U.
    :param ddpModel: DDP model
    :param ddpData: entired DDP data
    :param solverParams: DDP solver parameters
    """
    start = time.time()

    # Resetting convergence flag
    ddpData._convergence = False

    # Running an initial forward simulation. calculates the forward dynamics and
    # total costs
    Solver.forwardSimulation(ddpModel, ddpData)

    ddpData.muLM = solverParams.mu0LM
    ddpData.muV = solverParams.mu0V
    ddpData.alpha = solverParams.alpha0

    ddpData.n_iter = 0
    for i in xrange(solverParams.max_iter):
      # Recording the number of iterations
      ddpData.n_iter = i
      print ("Iteration", ddpData.n_iter, "muV", ddpData.muV,
             "muLM", ddpData.muLM, "alpha", ddpData.alpha)

      # Update the quadratic approximation
      Solver.updateQuadraticAppr(ddpModel, ddpData)

      while not Solver.backwardPass(ddpModel, ddpData, solverParams):
        # Quu is not positive-definitive, so increasing the
        # regularization factor
        if ddpData.muLM == 0.:
          ddpData.muLM = solverParams.mu0LM
        else:
          ddpData.muLM *= solverParams.muLM_inc
        print "\t", ("Quu isn't positive. Increasing muLM to", ddpData.muLM)
      print "\t","\t", "--------------------------------------------------Gradient Norm:", ddpData.gamma
      # Running the forward pass
      while not Solver.forwardPass(ddpModel, ddpData, solverParams):
        ddpData.alpha *= solverParams.alpha_dec
        print "\t", ("Rejected changes. Decreasing alpha to", ddpData.alpha)
        print "\t", "\t", "Reduction Ratio:", ddpData.z
        print "\t", "\t", "Expected Reduction:", ddpData.dV_exp
        print "\t", "\t", "Actual Reduction:", ddpData.dV
        if ddpData.alpha < solverParams.alpha_min:
          print "\t", ('It cannot be improved solution')
          ddpData._convergence = True
          break

      # Recording the total cost, gradient, weighted gradient (gamme),
      # regularization values and alpha for each iteration. This is useful for
      # analysing the solver performance
      if solverParams.record:
        solverParams.cost_itr[i] = ddpData.totalCost
        solverParams.gamma_itr[i] = ddpData.gamma
        solverParams.theta_itr[i] = ddpData.theta
        solverParams.muLM_itr[i] = ddpData.muLM
        solverParams.muV_itr[i] = ddpData.muV
        solverParams.alpha_itr[i] = ddpData.alpha

      # The quadratic model is accepted so for faster convergence it's better
      # to approach to Newton search direction. We can do it by decreasing the
      # Levenberg-Marquardt parameter
      ddpData.muLM *= solverParams.muLM_dec

      if ddpData.muLM < EPS: # this is full Newton direction
        ddpData.muLM = EPS

      # This regularization smooth the policy updates. Experimentally it helps
      # to reduce the number of iteration whenever the problem isn't well posed
      ddpData.muV *= solverParams.muV_dec
      if ddpData.muV < EPS:
        ddpData.muV = EPS

      # Increasing the stepsize for the next iteration
      ddpData.alpha *= solverParams.alpha_inc
      if ddpData.alpha > 1.:
        ddpData.alpha = 1.

      # Checking convergence
      if ddpData._convergence:
        # Final time
        end = time.time()
        print ("Reached convergence", ddpData.gamma, " in", end-start, "sec.")

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
  def forwardPass(ddpModel, ddpData, solverParams):
    """ Run the forward-pass of the DDP algorithm.

    The forward-pass basically applies a new policy and then rollout the
    system. After this rollouts, it's checked if this policy provides a
    reasonable improvement. For that we use Armijo condition to evaluated the
    choosen step length.
    :param ddpModel: DDP model
    :param ddpData: entired DDP data
    :param solverParams: DDP solver parameters
    """
    # Integrate the system along the new trajectory and compute its cost
    ddpData.totalCost = 0.
    for k in xrange(ddpData.N):
      # Getting the current DDP interval
      it = ddpData.intervalDataVector[k]

      # Computing the new control command
      np.copyto(it.u, it.u_prev +\
        ddpData.alpha * it.j + \
        np.dot(it.K, ddpModel.dynamicsModel.differenceState(it.dynamicsData,
                                                            it.x_prev,
                                                            it.x)))

      # Integrating the system dynamics and updating the new state value
      ddpModel.forwardRunningCalc(
        ddpData.intervalDataVector[k].dynamicsData,
        ddpData.intervalDataVector[k].costData,
        ddpData.intervalDataVector[k].x,
        ddpData.intervalDataVector[k].u,
        ddpData.intervalDataVector[k+1].x)
      ddpData.totalCost += ddpData.intervalDataVector[k].costData.l
    ddpModel.forwardTerminalCalc(
      ddpData.intervalDataVector[-1].dynamicsData,
      ddpData.intervalDataVector[-1].costData,
      ddpData.intervalDataVector[-1].x)
    ddpData.totalCost += ddpData.intervalDataVector[-1].costData.l

    # Checking convergence of the current iteration
    if np.abs(ddpData.gamma) <= solverParams.tol:
      ddpData._convergence = True
      return True

    # Checking the changes
    ddpData.dV = ddpData.totalCost_prev - ddpData.totalCost
    ddpData.z_new = ddpData.dV/ddpData.dV_exp
    if ddpData.z_new > solverParams.armijo_condition \
       and ddpData.z_new < solverParams.change_ub:
      ddpData.z = ddpData.z_new
      return True
    else:
      return False

  @staticmethod
  def backwardPass(ddpModel, ddpData, solverParams):
    """ Run the backward-pass of the DDP algorithm.

    The backward-pass is equivalent to a Riccati recursion. It updates the
    quadratic terms of the optimal control problem, and the gradient and
    Hessian of the value function. Additionally, it computes the new
    feedforward and feedback commands (i.e. control policy). A regularization
    scheme is used to ensure a good search direction. The norm of the gradient,
    a the directional derivatives are computed.
    :param ddpModel: DDP model
    :param ddpData: entired DDP data
    :param solverParams: DDP solver parameters
    """
    # Setting up the initial cost value, and the expected reduction equals zero
    ddpData.dV_exp = 0.

    # Running the backward sweep
    ddpData.gamma = 0.
    ddpData.theta = 0.
    np.copyto(ddpData.muI,
              ddpData.muLM * np.identity(ddpModel.dynamicsModel.nu()))

    np.copyto(ddpData.intervalDataVector[-1].Vx,
              ddpData.intervalDataVector[-1].costData.lx)
    np.copyto(ddpData.intervalDataVector[-1].Vxx,
              ddpData.intervalDataVector[-1].costData.lxx)
    for k in xrange(ddpData.N-1, -1, -1):
      it = ddpData.intervalDataVector[k]
      costData = it.costData
      dynamicsData = it.dynamicsData

      # Getting the value function values of the next interval (prime interval)
      fx = it.dynamicsData.discretizer.fx
      fu = it.dynamicsData.discretizer.fu

      # Updating the Q derivatives. Note that this is Gauss-Newton step because
      # we neglect the Hessian, it's also called iLQR.
      np.copyto(it.Vpxx_fx, np.dot(ddpData.intervalDataVector[k+1].Vxx, fx))
      np.copyto(it.Vpxx_fu, np.dot(ddpData.intervalDataVector[k+1].Vxx, fu))
      np.copyto(it.Qx, it.costData.lx + np.dot(fx.T, ddpData.intervalDataVector[k+1].Vx))
      np.copyto(it.Qu, it.costData.lu + np.dot(fu.T, ddpData.intervalDataVector[k+1].Vx))
      np.copyto(it.Quu, it.costData.luu +\
                np.dot(fu.T, it.Vpxx_fu))
      np.copyto(it.Qxx, it.costData.lxx +\
                np.dot(fx.T, it.Vpxx_fx))
      np.copyto(it.Qux, it.costData.lux +\
                np.dot(fu.T, it.Vpxx_fx))
      np.copyto(it.Quu_r, it.Quu + ddpData.muI + ddpData.muV * np.dot(fu.T, fu))
      if not isPositiveDefinitive(it.Quu_r, it.L):
        return False
      np.copyto(it.Qux_r, it.Qux + ddpData.muV * np.dot(fu.T, fx))

      # Computing the feedback and feedforward terms
      np.copyto(it.L_inv, np.linalg.inv(it.L))
      np.copyto(it.Quu_inv_minus, -1. * np.dot(it.L_inv.T, it.L_inv))
      np.copyto(it.K, np.dot(it.Quu_inv_minus, it.Qux_r))
      np.copyto(it.j, np.dot(it.Quu_inv_minus, it.Qu))

      # Computing the value function derivatives of this interval
      it.jt_Quu_j = 0.5 * np.asscalar(np.dot(it.j.T, np.dot(it.Quu, it.j)))
      it.jt_Qu = np.asscalar(np.dot(it.j.T, it.Qu))
      np.copyto(it.Vx, \
                it.Qx + np.dot(it.K.T, np.dot(it.Quu, it.j)) +\
                np.dot(it.K.T, it.Qu) + np.dot(it.Qux.T ,it.j))
      np.copyto(it.Vxx, \
                it.Qxx + np.dot(it.K.T, np.dot(it.Quu, it.K)) +\
                np.dot(it.K.T, it.Qux) + np.dot(it.Qux.T, it.K))

      # Symmetric can be lost due to round-off error. This ensures the symmetric
      np.copyto(it.Vxx, 0.5 * (it.Vxx + it.Vxx.T))

      # Updating the local cost and expected reduction. The total values are
      # used to check the changes in the forward pass. This method is explained
      # in Tassa's PhD thesis
      ddpData.dV_exp -= ddpData.alpha * (ddpData.alpha * it.jt_Quu_j + it.jt_Qu)

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
        ddpData.intervalDataVector[k].dynamicsData,
        ddpData.intervalDataVector[k].costData,
        ddpData.intervalDataVector[k].x,
        ddpData.intervalDataVector[k].u,
        ddpData.intervalDataVector[k+1].x)
      ddpData.totalCost += ddpData.intervalDataVector[k].costData.l
    ddpModel.forwardTerminalCalc(
      ddpData.intervalDataVector[-1].dynamicsData,
      ddpData.intervalDataVector[-1].costData,
      ddpData.intervalDataVector[-1].x)
    ddpData.totalCost += ddpData.intervalDataVector[-1].costData.l
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
      np.copyto(ddpData.intervalDataVector[k].x_prev,
                ddpData.intervalDataVector[k].x)
      np.copyto(ddpData.intervalDataVector[k].u_prev,
                ddpData.intervalDataVector[k].u)
      ddpModel.backwardRunningCalc(
        ddpData.intervalDataVector[k].dynamicsData,
        ddpData.intervalDataVector[k].costData,
        ddpData.intervalDataVector[k].x,
        ddpData.intervalDataVector[k].u)
    # Copying the current state into the previous one
    np.copyto(ddpData.intervalDataVector[-1].x_prev,
              ddpData.intervalDataVector[-1].x)
    ddpModel.backwardTerminalCalc(
      ddpData.intervalDataVector[-1].dynamicsData,
      ddpData.intervalDataVector[-1].costData,
      ddpData.intervalDataVector[-1].x)

  @staticmethod
  def getStateTrajectory(ddpData):
    """ Return the state trajectory contained in the ddpData.

    :param ddpData: entired DDP data
    """
    X = []
    for i in xrange(ddpData.N):
      X.append(ddpData.intervalDataVector[i].x)
    X.append(ddpData.intervalDataVector[-1].x)
    return X

  @staticmethod
  def getControlSequence(ddpData):
    """ Return the control sequence contained in the ddpData.

    :param ddpData: entired DDP data
    """
    U = []
    for i in xrange(ddpData.N):
      U.append(ddpData.intervalDataVector[i].u)
    return U

  @staticmethod
  def getFeedbackGainSequence(ddpData):
    """ Return the feedback gain sequences contained in the ddpData.

    :param ddpData: entired DDP data
    """
    K = []
    for i in xrange(ddpData.N):
      K.append(ddpData.intervalDataVector[i].K)
    return K

  @staticmethod
  def getFeedforwardSequence(ddpData):
    """ Return the feedforward control sequence contained in the ddpData.

    :param ddpData: entired DDP data
    """
    j = []
    for i in xrange(ddpData.N):
      j.append(ddpData.intervalDataVector[i].j)
    return j
