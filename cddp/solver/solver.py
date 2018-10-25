import time
from itertools import izip
import numpy as np
from cddp.utils import isPositiveDefinitive

class Solver(object):

  @staticmethod
  def setInitial(ddpModel, ddpData, xInit, UInit):
    """
    Performs data copying from init values to ddpData.
    """
    np.copyto(ddpData.intervalDataVector[0].dynamicsData.x, xInit)
    #np.copyto(ddpData.intervalDataVector[0].dynamicsData.x_new, xInit)
    for u, intervalData in izip(UInit,ddpData.intervalDataVector[:-1]):
      np.copyto(intervalData.dynamicsData.u, u)
    return

  @staticmethod
  def forwardSimulation(ddpModel, ddpData):
    """ Initial dynamics calculations, and cost calculations for one interval.
    This is one step of the forward pass of DDP.
    """
    for k in xrange(ddpData.N):
      ddpData.intervalDataVector[k].forwardCalc()
      ddpData.totalCost += ddpData.intervalDataVector[k].costData.l
      ddpModel.integrator(ddpModel, ddpData.intervalDataVector[k],
                          ddpData.intervalDataVector[k+1].dynamicsData.x)

    ddpData.intervalDataVector[-1].forwardCalc()
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
                np.dot(it.K, it.dynamicsData.deltaX(it.dynamicsData.x_prev,
                                                    it.dynamicsData.x)))

      # Integrating the system dynamics and updating the new state value
      ddpData.intervalDataVector[k].forwardCalc()
      ddpModel.integrator(ddpModel, ddpData.intervalDataVector[k],
                          ddpData.intervalDataVector[k+1].dynamicsData.x)
      ddpData.totalCost += ddpData.intervalDataVector[k].costData.l
    ddpData.intervalDataVector[-1].forwardCalc()
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
      itNext = ddpData.intervalDataVector[k+1]
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
      np.copyto(it.Qu, it.costData.lu +\
                fu.transposemultiply(ddpData.intervalDataVector[k+1].Vx))
      np.copyto(it.Qx, it.costData.lx +\
                fx.transposemultiplyarr(ddpData.intervalDataVector[k+1].Vx))
      np.copyto(it.Quu, it.costData.luu +\
                fu.transposemultiply(fu.premultiply(ddpData.intervalDataVector[k+1].Vxx)))
      np.copyto(it.Qxx, it.costData.lxx +\
                fx.transposemultiplymat(fx.premultiply(ddpData.intervalDataVector[k+1].Vxx)))
      np.copyto(it.Qux, it.costData.lux +\
                fu.transposemultiply(fx.premultiply(ddpData.intervalDataVector[k+1].Vxx)))

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
      np.copyto(it.Quu_r, it.Quu + ddpData.muI + np.dot(ddpData.muV, fu.square()))
      if not isPositiveDefinitive(it.Quu_r, it.L):
        return False
      np.copyto(it.Qux_r, it.Qux + ddpData.muV * fu.transposemultiply(fx()))

      # Computing the feedback and feedforward terms
      np.copyto(it.L_inv, np.linalg.inv(it.L))
      np.copyto(it.Quu_inv_minus, -1. * np.dot(it.L_inv.T, it.L_inv))
      np.copyto(it.K, np.dot(it.Quu_inv_minus, it.Qux_r))
      np.copyto(it.j, np.dot(it.Quu_inv_minus, it.Qu))

      # Computing the value function derivatives of this interval
      it.jt_Quu_j = 0.5 * (np.dot(it.j.T, np.dot(it.Quu, it.j)))[0,0]
      it.jt_Qu = np.dot(it.j.T, it.Qu)[0,0]
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
      ddpData.dV_exp += ddpData.alpha * (ddpData.alpha * it.jt_Quu_j + it.jt_Qu)

      # Updating the theta and gamma given the actual knot
      ddpData.gamma += np.dot(it.Qu.T, it.Qu)
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

    # Running an initial forward simulation
    # and calculates the forward dynamics and
    # total costs
    Solver.forwardSimulation(ddpModel, ddpData)

    # Resetting mu and alpha. As general rule of thumb we assume that the 
    # quadratic model has some error respect to the true model. So we start
    # closer to steeppest descent by adjusting the initial Levenberg-Marquardt
    # parameter (i.e. mu)
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
      for itData in ddpData.intervalDataVector:
        itData.backwardCalc()

      while not Solver.backwardPass(ddpModel, ddpData, solverParams):
        # Quu is not positive-definitive, so increasing the
        # regularization factor
        if ddpData.muLM == 0.:
          ddpData.muLM = solverParams.mu0LM
        else:
          ddpData.muLM *= solverParams.muLM_inc
        print "\t", ("Quu isn't positive. Increasing muLM to", ddpData.muLM)
      print "\t","\t", "--------------------------------------------------Expected Reduction:", -np.asscalar(ddpData.gamma)
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

      # This regularization smooth the policy updates. Experimentally it helps
      # to reduce the number of iteration whenever the problem isn't well posed
      self.muV *= self.muV_dec
      if self.muV < self.eps:
        self.muV = self.eps

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
    pass

  @staticmethod
  def calc():
    pass
