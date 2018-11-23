import unittest
import numpy as np
import cddp


class LinearDDPTest(unittest.TestCase):
  def setUp(self):
    # Creating the dynamic model of the system and its integrator
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    dynamics = cddp.SpringMass(integrator, discretizer)

    # Creating the cost manager and its cost functions
    costManager = cddp.CostManager()
    wx_term = 1e3 * np.ones((2 * dynamics.nv(),1))
    wx_track = 10. * np.ones((2 * dynamics.nv(),1))
    wu_reg = 1e-2 * np.ones((dynamics.nu(),1))
    xT_goal = cddp.StateCost(dynamics, wx_term)
    x_track = cddp.StateCost(dynamics, wx_track)
    u_reg = cddp.ControlCost(dynamics, wu_reg)

    # Adding the cost functions to the manager
    costManager.addTerminal(xT_goal, "xT_goal")
    costManager.addRunning(x_track, "x_track")
    costManager.addRunning(u_reg, "u_reg")

    # Setting up the DDP problem
    timeline = np.arange(0.0, 0.25, 1e-3)
    self.ddpModel = cddp.DDPModel(dynamics, costManager)
    self.ddpData = cddp.DDPData(self.ddpModel, timeline)

    # Setting up the initial conditions
    # Create random initial
    x0 = np.random.rand(dynamics.nx(), 1)
    u0 = np.zeros((dynamics.nu(),1))
    U0 = [u0 for i in xrange(len(timeline)-1)]
    self.ddpModel.setInitial(self.ddpData, xInit=x0, UInit=U0)

    # Setting up the desired reference for each single cost function
    xref = np.array([ [1.],[0.] ])
    Xref = [xref for i in xrange(len(timeline))]
    self.ddpModel.setRunningReference(self.ddpData, Xref[:-1], "x_track")
    self.ddpModel.setTerminalReference(self.ddpData, Xref[-1], "xT_goal")

  def test_problem_solved_in_one_iteration(self):
    # Using the default solver parameters
    solverParams = cddp.SolverParams()

    # Running the DDP solver
    cddp.Solver.solve(self.ddpModel, self.ddpData, solverParams)
    self.assertEqual(
      self.ddpData.n_iter, 1, \
      "This is equivalent to a LQR problem which it solves in 1 iteration.")

  def test_improvement_ratio_equals_one(self):
    # Using the default solver parameters
    solverParams = cddp.SolverParams()

    # Running the DDP solver
    cddp.Solver.solve(self.ddpModel, self.ddpData, solverParams)
    print  self.ddpData.z
    self.assertAlmostEqual(
      self.ddpData.z, 1., 1, \
      "This is a LQR problem the improvement ration should be equals to 1.")


  # def test_against_kkt_solution(self):
  #   # Running the DDP solver
  #   self.ddp.compute(self.x0)

  #   # Creating the variables of the KKT problem
  #   n = self.ddp.intervals[0].system.ndx
  #   m = self.ddp.intervals[0].system.m
  #   N = self.ddp.N
  #   G = np.matrix(np.zeros((N*n+n, 1)))
  #   gradJ = np.matrix(np.zeros((N*(n+m)+n, 1)))
  #   gradG = np.matrix(np.zeros((N*(n+m)+n, N*n+n)))
  #   hessL = np.matrix(np.zeros((N*(n+m)+n, N*(n+m)+n)))

  #   # Running a backward-pass in order to update the derivatives of the
  #   # DDP solution (no regularization and full Newton step)
  #   self.ddp.backwardPass(0., 0., 1.)

  #   # Building the KKT matrix and vector given the cost and dynamics derivatives
  #   # from the DDP backward-pass
  #   for k in range(N):
  #     data = self.ddp.intervals[k]

  #     # Running cost and its derivatives
  #     lx = data.cost.total.lx
  #     lu = data.cost.total.lu
  #     lxx = data.cost.total.lxx
  #     luu = data.cost.total.luu
  #     lux = data.cost.total.lux

  #     # Dynamics and its derivatives
  #     f = data.system.f
  #     fx = data.system.fx
  #     fu = data.system.fu

  #     # Updating the constraint and cost functions and them gradient, and the
  #     # Hessian of this problem
  #     G[k*n:(k+1)*n] = f
  #     gradG[k*(n+m):(k+1)*(n+m), k*n:(k+1)*n] = np.block([ [fx.T],[fu.T] ])
  #     gradJ[k*(n+m):(k+1)*(n+m)] = np.block([ [lx],[lu] ])
  #     hessL[k*(n+m):(k+1)*(n+m),k*(n+m):(k+1)*(n+m)] = \
  #       np.block([ [lxx, lux.T],[lux, luu] ])

  #   # Updating the terms given the terminal state
  #   G[N*n:(N+1)*n] = f
  #   gradG[N*(n+m):(N+1)*(n+m), N*n:(N+1)*n] = fx.T
  #   gradJ[N*(n+m):(N+1)*(n+m)] = lx
  #   hessL[N*(n+m):(N+1)*(n+m), N*(n+m):(N+1)*(n+m)] = \
  #     self.ddp.terminal_interval.cost.total.lxx

  #   # Computing the KKT matrix and vector
  #   kkt_mat = np.block([ [hessL,gradG],[gradG.T, np.zeros((N*n+n,N*n+n))] ])
  #   kkt_vec = np.block([ [gradJ],[G] ])

  #   # Solving the KKT problem
  #   sol = np.linalg.solve(kkt_mat, kkt_vec)

  #   # Recording the KKT solution into a list
  #   X_kkt = []
  #   U_kkt = []
  #   for k in range(N):
  #     w = sol[k*(n+m):(k+1)*(n+m)]
  #     X_kkt.append(w[:n])
  #     U_kkt.append(w[-m])

  #   # Getting the DDP solution
  #   X_opt = self.ddp.getStateTrajectory()
  #   U_opt = self.ddp.getControlSequence()

  #   for i in range(N-1):
  #     # Checking the DDP solution is almost equals to KKT solution
  #     self.assertTrue(np.allclose(X_kkt[i], X_opt[i], atol=1e-3),
  #                            "State KKT solution at " + str(i) + " is not the same.")
  #     # self.assertTrue(np.allclose(U_kkt[i], U_opt[i], atol=1e-2),
  #     #   "Control KKT solution at " + str(i) + " is not the same.")

  # def test_positive_expected_improvement(self):
  #   # Running the DDP solver
  #   self.ddp.compute(self.x0)
  #   self.assertGreater(-self.ddp.dV_exp, 0.,
  #                      "The expected improvement is not positive.")

  # def test_regularization_in_backward_pass(self):
  #   # Backward-pass without regularization
  #   self.ddp.backwardPass(0., 0., 1.)
  #   Vxx = self.ddp.intervals[-1].Vxx.copy()
  #   # Backward-pass with regularization
  #   mu = np.random.random_sample()
  #   self.ddp.backwardPass(mu, 0., 1.)
  #   Vxx_reg = self.ddp.intervals[-1].Vxx.copy()
  #   self.assertTrue(np.allclose(Vxx, Vxx_reg),
  #                       "Regularization doesn't affect the terminal Vxx.")


if __name__ == '__main__':
  unittest.main()