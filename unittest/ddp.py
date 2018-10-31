import unittest
import numpy as np
import cddp


class LinearDDPTest(unittest.TestCase):
  def setUp(self):
    # Creating the dynamic model of the system and its integrator
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    system = cddp.SpringMass(integrator, discretizer)

    # Create random initial and desired state
    self.x0 = np.random.rand(system.getConfigurationDimension(), 1)
    x_des = np.random.rand(system.getConfigurationDimension(), 1)
    x_des[1] = 0.

    # Creating the cost manager and its cost functions
    cost_manager = cddp.CostManager()
    goal_cost = cddp.StateResidualTerminalQuadraticCost(x_des)
    xu_cost = cddp.StateControlRunningQuadraticCost(x_des)

    # Setting up the weights of the quadratic terms
    wx = 100.* np.ones(system.getConfigurationDimension())
    wu = 0.001 * np.ones(system.getControlDimension())
    goal_cost.setWeights(0.5 * wx)
    xu_cost.setWeights(wx, wu)

    # Adding the cost functions to the manager
    cost_manager.addTerminal(goal_cost)
    cost_manager.addRunning(xu_cost)

    # Creating the DDP solver
    timeline = np.arange(0.0, 0.5, 0.001)  # np.linspace(0., 0.5, 51)
    self.ddp = cddp.DDP(system, cost_manager, timeline)

  def test_against_kkt_solution(self):
    # Running the DDP solver
    self.ddp.compute(self.x0)

    # Creating the variables of the KKT problem
    n = self.ddp.intervals[0].system.ndx
    m = self.ddp.intervals[0].system.m
    N = self.ddp.N
    G = np.matrix(np.zeros((N*n+n, 1)))
    gradJ = np.matrix(np.zeros((N*(n+m)+n, 1)))
    gradG = np.matrix(np.zeros((N*(n+m)+n, N*n+n)))
    hessL = np.matrix(np.zeros((N*(n+m)+n, N*(n+m)+n)))

    # Running a backward-pass in order to update the derivatives of the
    # DDP solution (no regularization and full Newton step)
    self.ddp.backwardPass(0., 0., 1.)

    # Building the KKT matrix and vector given the cost and dynamics derivatives
    # from the DDP backward-pass
    for k in range(N):
      data = self.ddp.intervals[k]

      # Running cost and its derivatives
      lx = data.cost.total.lx
      lu = data.cost.total.lu
      lxx = data.cost.total.lxx
      luu = data.cost.total.luu
      lux = data.cost.total.lux

      # Dynamics and its derivatives
      f = data.system.f
      fx = data.system.fx
      fu = data.system.fu

      # Updating the constraint and cost functions and them gradient, and the
      # Hessian of this problem
      G[k*n:(k+1)*n] = f
      gradG[k*(n+m):(k+1)*(n+m), k*n:(k+1)*n] = np.block([ [fx.T],[fu.T] ])
      gradJ[k*(n+m):(k+1)*(n+m)] = np.block([ [lx],[lu] ])
      hessL[k*(n+m):(k+1)*(n+m),k*(n+m):(k+1)*(n+m)] = \
        np.block([ [lxx, lux.T],[lux, luu] ])

    # Updating the terms given the terminal state
    G[N*n:(N+1)*n] = f
    gradG[N*(n+m):(N+1)*(n+m), N*n:(N+1)*n] = fx.T
    gradJ[N*(n+m):(N+1)*(n+m)] = lx
    hessL[N*(n+m):(N+1)*(n+m), N*(n+m):(N+1)*(n+m)] = \
      self.ddp.terminal_interval.cost.total.lxx

    # Computing the KKT matrix and vector
    kkt_mat = np.block([ [hessL,gradG],[gradG.T, np.zeros((N*n+n,N*n+n))] ])
    kkt_vec = np.block([ [gradJ],[G] ])

    # Solving the KKT problem
    sol = np.linalg.solve(kkt_mat, kkt_vec)

    # Recording the KKT solution into a list
    X_kkt = []
    U_kkt = []
    for k in range(N):
      w = sol[k*(n+m):(k+1)*(n+m)]
      X_kkt.append(w[:n])
      U_kkt.append(w[-m])

    # Getting the DDP solution
    X_opt = self.ddp.getStateTrajectory()
    U_opt = self.ddp.getControlSequence()

    for i in range(N-1):
      # Checking the DDP solution is almost equals to KKT solution
      self.assertTrue(np.allclose(X_kkt[i], X_opt[i], atol=1e-3),
                             "State KKT solution at " + str(i) + " is not the same.")
      # self.assertTrue(np.allclose(U_kkt[i], U_opt[i], atol=1e-2),
      #   "Control KKT solution at " + str(i) + " is not the same.")

  def test_positive_expected_improvement(self):
    # Running the DDP solver
    self.ddp.compute(self.x0)
    self.assertGreater(-self.ddp.dV_exp, 0.,
                       "The expected improvement is not positive.")

  def test_positive_obtained_improvement(self):
    # Running the DDP solver
    self.ddp.compute(self.x0)
    self.assertGreater(-self.ddp.dV, 0.,
                       "The obtained improvement is not positive.")

  def test_improvement_ratio_equals_one(self):
    # Running the DDP solver
    self.ddp.compute(self.x0)
    self.assertAlmostEqual(
      self.ddp.z, 1., 2, \
      "The improvement ration is not equals to 1.")

  def test_regularization_in_backward_pass(self):
    # Backward-pass without regularization
    self.ddp.backwardPass(0., 0., 1.)
    Vxx = self.ddp.intervals[-1].Vxx.copy()
    # Backward-pass with regularization
    mu = np.random.random_sample()
    self.ddp.backwardPass(mu, 0., 1.)
    Vxx_reg = self.ddp.intervals[-1].Vxx.copy()
    self.assertTrue(np.allclose(Vxx, Vxx_reg),
                        "Regularization doesn't affect the terminal Vxx.")


if __name__ == '__main__':
  unittest.main()