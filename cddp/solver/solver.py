import time
import numpy as np
from cddp.utils import isPositiveDefinitive, EPS


class Solver(object):

  @staticmethod
  def forwardSimulation(ddpModel, ddpData):
    """ Initial dynamics calculations, and cost calculations for one interval.
    This is one step of the forward pass of DDP.
    """
    for k in xrange(ddpData.N):
      ddpModel.forwardRunningCalc(ddpData.intervalDataVector[k])
      ddpData.totalCost += ddpData.intervalDataVector[k].costData.l
      ddpModel.integrator(ddpModel, ddpData.intervalDataVector[k],
                          ddpData.intervalDataVector[k+1].dynamicsData.x)

    ddpModel.forwardTerminalCalc(ddpData.intervalDataVector[-1])
    ddpData.totalCost += ddpData.intervalDataVector[-1].costData.l
    return

  @staticmethod
  def forwardPass(ddpModel, ddpData, solverParams):
    """ Runs the forward pass of the DDP algorithm.
    """
    ddpData.totalCost = 0.
    # Integrate the system along the new trajectory
    for k in xrange(ddpData.N):
      # Getting the current DDP interval
      it = ddpData.intervalDataVector[k]
      itNext = ddpData.intervalDataVector[k]
      # Computing the new control command
      np.copyto(it.dynamicsData.u, it.dynamicsData.u_prev +\
                ddpData.alpha * it.j +\
                np.dot(it.K, ddpModel.dynamicsModel.deltaX(it.dynamicsData,
                                                           it.dynamicsData.x_prev,
                                                           it.dynamicsData.x)))

      # Integrating the system dynamics and updating the new state value
      ddpModel.forwardRunningCalc(ddpData.intervalDataVector[k])
      ddpModel.integrator(ddpModel, ddpData.intervalDataVector[k],
                          ddpData.intervalDataVector[k+1].dynamicsData.x)
      ddpData.totalCost += ddpData.intervalDataVector[k].costData.l
    ddpModel.forwardTerminalCalc(ddpData.intervalDataVector[-1])
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
    """ Runs the backward pass of the DDP algorithm.
    """

    # Setting up the initial cost value, and the expected reduction equals zero
    ddpData.dV_exp = 0.

    # Running the backward sweep
    ddpData.gamma = 0.
    ddpData.theta = 0.
    np.copyto(ddpData.muI,
              ddpData.muLM * np.identity(ddpModel.dynamicsModel.nu()))

    np.copyto(ddpData.intervalDataVector[-1].Vx, ddpData.intervalDataVector[-1].costData.lx)
    np.copyto(ddpData.intervalDataVector[-1].Vxx,ddpData.intervalDataVector[-1].costData.lxx)
    for k in range(ddpData.N-1, -1, -1):
      it = ddpData.intervalDataVector[k]
      costData = it.costData
      dynamicsData = it.dynamicsData

      # Getting the state, control and step time of the interval
      x = dynamicsData.x
      u = dynamicsData.u
      dt = it.dt

      # Getting the value function values of the next interval (prime interval)
      fx = it.dynamicsData.fx
      fu = it.dynamicsData.fu
      # Updating the Q derivatives. Note that this is Gauss-Newton step because
      # we neglect the Hessian, it's also called iLQR.

      #TODO: UNCOMMENT AFTER DEBUG
      np.copyto(it.Qx, it.costData.lx +\
                fx.transposemultiplyarr(ddpData.intervalDataVector[k+1].Vx))
      np.copyto(it.Qu, it.costData.lu +\
                fu.transposemultiply(ddpData.intervalDataVector[k+1].Vx))
      np.copyto(it.Quu, it.costData.luu +\
                fu.transposemultiply(fu.premultiply(ddpData.intervalDataVector[k+1].Vxx)))
      np.copyto(it.Qxx, it.costData.lxx +\
                fx.transposemultiplymat(fx.premultiply(ddpData.intervalDataVector[k+1].Vxx)))
      np.copyto(it.Qux, it.costData.lux +\
                fu.transposemultiply(fx.premultiply(ddpData.intervalDataVector[k+1].Vxx)))
      np.copyto(it.Quu_r, it.Quu + ddpData.muI + np.dot(ddpData.muV, fu.square()))
      if not isPositiveDefinitive(it.Quu_r, it.L):
        return False
      np.copyto(it.Qux_r, it.Qux + ddpData.muV * fu.transposemultiply(fx()))

      ########################################################################
      
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
  def solve(ddpModel, ddpData, solverParams):
    """ Conducts the DDP algorithm.
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
    for i in range(solverParams.max_iter):
      # Recording the number of iterations
      ddpData.n_iter = i
      print ("Iteration", ddpData.n_iter, "muV", ddpData.muV,
             "muLM", ddpData.muLM, "alpha", ddpData.alpha)

      # Prepare for DDP Backward Pass. TODO: Parallelize.
      for k in xrange(ddpData.N):
        ddpModel.backwardRunningCalc(ddpData.intervalDataVector[k])
      ddpModel.backwardTerminalCalc(ddpData.intervalDataVector[-1])

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

      # Recording the regularization values, gradient, and theta for each
      # iteration. This is useful for analysing the solver performance

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
  def calc():
    pass
