import time
from itertools import izip
import numpy as np

class Solver(object):

  @staticmethod
  def setInitial(ddpModel, ddpData, xInit, UInit):
    """
    Performs data copying from init values to ddpData.
    """
    np.copyto(ddpData.intervalDataVector[0].dynamicsData.x, xInit)
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
      ddpModel.integrator(ddpModel, ddpData.intervalDataVector[k],
                          ddpData.intervalDataVector[k+1].dynamicsData.x)
    ddpData.intervalDataVector[-1].forwardCalc()
    return

  @staticmethod
  def solve(ddpModel, ddpData, solverParams):
    """ Conducts the DDP algorithm.
    """
    start = time.time()

    # Resetting convergence flag
    ddpData._convergence = False

    # Running an initial forward simulation given the initial state and
    # the control sequence
    Solver.forwardSimulation(ddpModel, ddpData)

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
