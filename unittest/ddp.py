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
    # Running the DDP solver and updating its derivatives
    solverParams = cddp.SolverParams()
    cddp.Solver.solve(self.ddpModel, self.ddpData, solverParams)
    cddp.Solver.updateQuadraticAppr(self.ddpModel, self.ddpData)

    nx = self.ddpModel.dynamicsModel.nx()
    nu = self.ddpModel.dynamicsModel.nu()
    N = self.ddpData.N
    sol, lag, hess, jac, grad, g = self.KKTSolver(self.ddpModel, self.ddpData)

    # Recording the KKT solution into a list
    X_kkt = []
    U_kkt = []
    for k in range(N):
      w = sol[k*(nx+nu):(k+1)*(nx+nu)]
      X_kkt.append(w[:nx])
      U_kkt.append(w[-nu])
    X_kkt.append(sol[-nx:])

    for i in range(N):
      # Checking that the KKT solution is equals to the DDP solution. Since we
      # passed the DDP solution to the KKT solver, then we expected that the
      # KKT solution is equals to zero
      self.assertTrue(np.allclose(X_kkt[i], 0.),
        "Delta state from KKT solution should be equals zero.")
      self.assertTrue(np.allclose(U_kkt[i], 0.,),
        "Delta control from KKT solution should be equals zero.")
    self.assertTrue(np.allclose(X_kkt[N], 0.),
        "Delta state from KKT solution should be equals zero.")

    # Checking that the Vx is equals to the KKT Lagrangian multipliers
    for i in range(N+1):
      Vx = self.ddpData.intervalDataVector[i].Vx
      self.assertTrue(np.allclose(Vx, lag[i*nx:(i+1)*nx]),
      "Vx should be equals to the Lagrangian multiplier.")

    # Checking that the KKT Lagrangian norm is equals to zero. This checks the
    # DDP stopping criteria
    lagrangian = grad + np.dot(jac.T, lag)
    self.assertAlmostEqual(
      np.linalg.norm(lagrangian), 0., 7, \
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

  def KKTSolver(self, ddpModel, ddpData):
    # Generate a warm-point trajectory from an initial condition
    # cddp.Solver.forwardSimulation(ddpModel, ddpData)

    # Compute the derivatives along a warm-point trajectory
    # cddp.Solver.updateQuadraticAppr(ddpModel, ddpData)

    # Creating the variables of the KKT problem
    nx = ddpModel.dynamicsModel.nx()
    nu = ddpModel.dynamicsModel.nu()
    N = ddpData.N
    nw = nx + nu
    nvar = N * nw + nx
    ncon = N * nx + nx
    hess = np.matrix(np.zeros((nvar,nvar)))
    jac = np.matrix(np.zeros((ncon,nvar)))
    grad = np.matrix(np.zeros((nvar,1)))
    g = np.matrix(np.zeros((ncon,1)))
    
    # Building the KKT matrix and vector given the cost and dynamics derivatives
    # from the DDP backward-pass
    Ixx = np.eye(nx)
    Oxx = np.zeros((nx,nx))
    Oxu = np.zeros((nx,nu))
    g[:nx] = ddpData.intervalDataVector[0].dynamicsData.x
    for k in range(N):
      # Interval data
      data = ddpData.intervalDataVector[k]

      # State, control and decision vector
      x_i = data.dynamicsData.x
      u_i = data.dynamicsData.u
      w_i = np.block([ [x_i],[u_i] ])

      # Running cost and its derivatives
      lx_i = data.costData.lx
      lu_i = data.costData.lu
      lxx_i = data.costData.lxx
      luu_i = data.costData.luu
      lux_i = data.costData.lux
      q_i = np.block([ [lx_i],[lu_i] ])
      Q_i = np.block([ [lxx_i, lux_i.T],[lux_i, luu_i] ])

      # Dynamics and its derivatives  
      fx_i = data.dynamicsData.discretizer.fx
      fu_i = data.dynamicsData.discretizer.fu
      f_i = np.block([fx_i, fu_i])

      # Updating the constraint and cost functions and their gradient, and the
      # Hessian of this problem
      hess[k*nw:(k+1)*nw, k*nw:(k+1)*nw] = Q_i
      jac[k*nx:(k+2)*nx, k*nw:(k+1)*nw] = np.block([ [-Ixx, Oxu],[f_i] ])
      grad[k*nw:(k+1)*nw] = q_i
      g[k*nx:(k+2)*nx] += np.dot(np.block([ [-Ixx, Oxu],[ f_i] ]), w_i)

    # Terminal state and cost derivatives
    x_T = ddpData.intervalDataVector[-1].dynamicsData.x
    lx_T = ddpData.intervalDataVector[-1].costData.lx
    lxx_T = ddpData.intervalDataVector[-1].costData.lxx

    # Updating the terms regarding the terminal condition
    hess[N*nw:N*nw+nx, N*nw:N*nw+nx] = lxx_T
    jac[N*nx:(N+2)*nx, N*nw:N*nw+nx] = -Ixx
    grad[N*nw:(N+1)*nw] = lx_T
    g[N*nx:(N+2)*nx] += -x_T

    # Computing the KKT matrix and its vector
    kkt_mat = np.block([ [hess, jac.T],[jac, np.zeros((ncon,ncon))] ])
    kkt_vec = np.block([ [grad],[g] ])

    # Solving the KKT problem
    sol = np.linalg.solve(kkt_mat, -kkt_vec)
    w = sol[:nvar]
    lag = sol[nvar:]
    return w, lag, hess, jac, grad, g

if __name__ == '__main__':
  unittest.main()