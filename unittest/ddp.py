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
    self.assertAlmostEqual(
      self.ddpData.z, 1., 1, \
      "This is a LQR problem the improvement ration should be equals to 1.")

  def test_against_kkt_solution(self):
    # Using the default solver parameters
    solverParams = cddp.SolverParams()

    # Running the DDP solver
    cddp.Solver.solve(self.ddpModel, self.ddpData, solverParams)

    n = self.ddpModel.dynamicsModel.nx()
    m = self.ddpModel.dynamicsModel.nu()
    N = self.ddpData.N
    hessL, gradG, gradJ, G = self.buildKKTProblem(self.ddpModel, self.ddpData)
    sol, lag = \
      self.computeKKTPoint(self.ddpModel, self.ddpData, hessL, gradG, gradJ, G)

    # Recording the KKT solution into a list
    X_kkt = []
    U_kkt = []
    for k in range(N):
      w = sol[k*(n+m):(k+1)*(n+m)]
      X_kkt.append(w[:n])
      U_kkt.append(w[-m])

    # Getting the DDP solution
    X_opt = cddp.Solver.getStateTrajectory(self.ddpData)
    U_opt = cddp.Solver.getControlSequence(self.ddpData)

    for i in range(N):
      # Checking the DDP solution is almost equals to KKT solution
      self.assertTrue(np.allclose(X_kkt[i], X_opt[i], atol=1e-3),
        "State KKT solution at " + str(i) + " is not the same.")
      self.assertTrue(np.allclose(U_kkt[i], U_opt[i], atol=1e-2),
        "Control KKT solution at " + str(i) + " is not the same.")

  def test_stopping_criteria_against_kkt(self):
    # Using the default solver parameters
    solverParams = cddp.SolverParams()

    # Running the DDP solver
    cddp.Solver.solve(self.ddpModel, self.ddpData, solverParams)

    n = self.ddpModel.dynamicsModel.nx()
    m = self.ddpModel.dynamicsModel.nu()
    N = self.ddpData.N
    hessL, gradG, gradJ, G = self.buildKKTProblem(self.ddpModel, self.ddpData)
    sol, lag = \
      self.computeKKTPoint(self.ddpModel, self.ddpData, hessL, gradG, gradJ, G)

    self.assertAlmostEqual(
      np.linalg.norm(hessL * sol + gradG * lag - gradJ), 0., 7, \
      "The Lagrangian gradient computed from the KKT solution is not equals zero.")

  def test_regularization_in_backward_pass(self):
    # Using the default solver parameters
    solverParams = cddp.SolverParams()

    # Running a backward-pass without regularization
    self.ddpData.muV = 0.
    self.ddpData.muLM = 0.
    self.ddpData.alpha = 1.
    cddp.Solver.updateQuadraticAppr(self.ddpModel, self.ddpData)
    cddp.Solver.backwardPass(self.ddpModel, self.ddpData, solverParams)
    Vxx = self.ddpData.intervalDataVector[-1].Vxx.copy()

    # Backward-pass with regularization
    self.ddpData.muLM = np.random.random_sample()
    cddp.Solver.updateQuadraticAppr(self.ddpModel, self.ddpData)
    cddp.Solver.backwardPass(self.ddpModel, self.ddpData, solverParams)
    Vxx_reg = self.ddpData.intervalDataVector[-1].Vxx.copy()

    # Checking both values of the Value-function Hessian
    self.assertTrue(np.allclose(Vxx, Vxx_reg),
                        "Regularization doesn't affect the terminal Vxx.")

  def buildKKTProblem(self, ddpModel, ddpData):
    # Creating the variables of the KKT problem
    n = ddpModel.dynamicsModel.nx()
    m = ddpModel.dynamicsModel.nu()
    N = ddpData.N
    nvar = N*(n+m)+n
    ncon = N*n+n
    G = np.matrix(np.zeros((ncon,1)))
    gradJ = np.matrix(np.zeros((nvar,1)))
    gradG = np.matrix(np.zeros((nvar,ncon)))
    hessL = np.matrix(np.zeros((nvar,nvar)))

    # Building the KKT matrix and vector given the cost and dynamics derivatives
    # from the DDP backward-pass
    for k in range(N):
      data = ddpData.intervalDataVector[k]

      # Running cost and its derivatives
      lx = data.costData.lx
      lu = data.costData.lu
      lxx = data.costData.lxx
      luu = data.costData.luu
      lux = data.costData.lux

      # Dynamics and its derivatives
      f = ddpData.intervalDataVector[k+1].dynamicsData.x
      fx = data.dynamicsData.discretizer.fx
      fu = data.dynamicsData.discretizer.fu

      # Updating the constraint and cost functions and them gradient, and the
      # Hessian of this problem
      G[k*n:(k+1)*n] = f
      gradG[k*(n+m):(k+1)*(n+m), k*n:(k+1)*n] = np.block([ [fx.T],[fu.T] ])
      gradJ[k*(n+m):(k+1)*(n+m)] = np.block([ [lx],[lu] ])
      hessL[k*(n+m):(k+1)*(n+m), k*(n+m):(k+1)*(n+m)] = \
        np.block([ [lxx, lux.T],[lux, luu] ])

    # Updating the terms given the terminal state
    G[N*n:(N+1)*n] = f
    gradG[N*(n+m):(N+1)*(n+m), N*n:(N+1)*n] = np.eye(n)
    gradJ[N*(n+m):(N+1)*(n+m)] = \
      ddpData.intervalDataVector[-1].costData.lx
    hessL[N*(n+m):(N+1)*(n+m), N*(n+m):(N+1)*(n+m)] = \
      ddpData.intervalDataVector[-1].costData.lxx
    return hessL, gradG, gradJ, G

  def computeKKTPoint(self, ddpModel, ddpData, hessL, gradG, gradJ, G):
    n = ddpModel.dynamicsModel.nx()
    m = ddpModel.dynamicsModel.nu()
    N = ddpData.N

    # Computing the KKT matrix and vector
    kkt_mat = np.block([ [hessL,gradG],[gradG.T, np.zeros((N*n+n,N*n+n))] ])
    kkt_vec = np.block([ [gradJ],[G] ])

    # Solving the KKT problem
    sol = np.linalg.solve(kkt_mat, kkt_vec)
    w = sol[:(N+1)*(n+m)-m]
    lag = sol[(N+1)*(n+m)-m:]
    return w, lag

if __name__ == '__main__':
  unittest.main()