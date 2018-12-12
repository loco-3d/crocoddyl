import unittest
import numpy as np
import crocoddyL


class LinearDDPTest(unittest.TestCase):
  def setUp(self):
    # Creating the dynamic model of the system and its integrator
    integrator = crocoddyL.EulerIntegrator()
    discretizer = crocoddyL.EulerDiscretizer()
    dynamics = crocoddyL.SpringMass(integrator, discretizer)

    # Creating the cost manager and its cost functions
    costManager = crocoddyL.CostManager()
    wx_term = 1e3 * np.ones((2 * dynamics.nv(),1))
    wx_track = 10. * np.ones((2 * dynamics.nv(),1))
    wu_reg = 1e-2 * np.ones((dynamics.nu(),1))
    xT_goal = crocoddyL.StateCost(dynamics, wx_term)
    x_track = crocoddyL.StateCost(dynamics, wx_track)
    u_reg = crocoddyL.ControlCost(dynamics, wu_reg)

    # Adding the cost functions to the manager
    costManager.addTerminal(xT_goal, "xT_goal")
    costManager.addRunning(x_track, "x_track")
    costManager.addRunning(u_reg, "u_reg")

    # Setting up the DDP problem
    timeline = np.arange(0.0, 0.25, 1e-3)
    self.ddpModel = crocoddyL.DDPModel(dynamics, costManager)
    self.ddpData = crocoddyL.DDPData(self.ddpModel, timeline)

    # Setting up the initial conditions
    # Create random initial
    x0 = np.random.rand(dynamics.nx(), 1)
    u0 = np.zeros((dynamics.nu(),1))
    U0 = [u0 for i in xrange(self.ddpData.N)]
    self.ddpModel.setInitial(self.ddpData, xInit=x0, UInit=U0)

    # Setting up the desired reference for each single cost function
    xref = np.array([ [1.],[0.] ])
    Xref = [xref for i in xrange(self.ddpData.N+1)]
    self.ddpModel.setRunningReference(self.ddpData, Xref[:-1], "x_track")
    self.ddpModel.setTerminalReference(self.ddpData, Xref[-1], "xT_goal")

  def test_problem_solved_in_one_iteration(self):
    # Using the default solver parameters
    solverParams = crocoddyL.SolverParams()

    # Running the DDP solver
    crocoddyL.Solver.solve(self.ddpModel, self.ddpData, solverParams)
    self.assertEqual(
      self.ddpData.n_iter, 1, \
      "This is equivalent to a LQR problem which it solves in 1 iteration.")

  def test_improvement_ratio_equals_one(self):
    # Using the default solver parameters
    solverParams = crocoddyL.SolverParams()

    # Running the DDP solver
    crocoddyL.Solver.solve(self.ddpModel, self.ddpData, solverParams)
    self.assertAlmostEqual(
      self.ddpData.z, 1., 1, \
      "This is a LQR problem the improvement ration should be equals to 1.")

  def test_against_kkt_solution(self):
    # Running the DDP solver and updating its derivatives
    solverParams = crocoddyL.SolverParams()
    crocoddyL.Solver.solve(self.ddpModel, self.ddpData, solverParams)
    crocoddyL.Solver.updateQuadraticAppr(self.ddpModel, self.ddpData)

    nx = self.ddpModel.dynamicModel.nx()
    nu = self.ddpModel.dynamicModel.nu()
    N = self.ddpData.N
    sol, lag, hess, jac, grad, g = self.KKTSolver(self.ddpModel, self.ddpData)

    # Recording the KKT solution into a list
    X_kkt = []
    U_kkt = []
    for k in xrange(N):
      w = sol[k*(nx+nu):(k+1)*(nx+nu)]
      X_kkt.append(w[:nx])
      U_kkt.append(w[-nu])
    X_kkt.append(sol[-nx:])

    for i in xrange(N):
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
    for i in xrange(N+1):
      Vx = self.ddpData.interval[i].Vx
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
    solverParams = crocoddyL.SolverParams()

    # Running a backward-pass with control regularization
    self.ddpData.muLM = 0.
    self.ddpData.muV = 0.
    self.ddpData.alpha = 1.
    crocoddyL.Solver.forwardSimulation(self.ddpModel, self.ddpData)
    crocoddyL.Solver.updateQuadraticAppr(self.ddpModel, self.ddpData)
    crocoddyL.Solver.backwardPass(self.ddpModel, self.ddpData, solverParams)

    # Recording the Vx values for the control regularization case
    Vx_ctrl = []
    for i in xrange(self.ddpData.N + 1):
      Vx_ctrl.append(self.ddpData.interval[i].Vx.copy())

    # Running a backward-pass with an equivalente LM regularization
    self.ddpData.muLM = 1e-2
    self.ddpModel.costManager.removeRunning('u_reg')
    crocoddyL.Solver.forwardSimulation(self.ddpModel, self.ddpData)
    crocoddyL.Solver.updateQuadraticAppr(self.ddpModel, self.ddpData)
    crocoddyL.Solver.backwardPass(self.ddpModel, self.ddpData, solverParams)

    # Recording the Vx values for the control regularization case
    Vx_reg = []
    for i in xrange(self.ddpData.N + 1):
      Vx_reg.append(self.ddpData.interval[i].Vx.copy())

    # Checking
    for i in xrange(self.ddpData.N + 1):
      self.assertTrue(np.allclose(Vx_ctrl[i], Vx_reg[i]),
        "The control cost regularization isn't equal to the LM regularization.")

  def KKTSolver(self, ddpModel, ddpData):
    # Generate a warm-point trajectory from an initial condition
    # crocoddyL.Solver.forwardSimulation(ddpModel, ddpData)

    # Compute the derivatives along a warm-point trajectory
    # crocoddyL.Solver.updateQuadraticAppr(ddpModel, ddpData)

    # Creating the variables of the KKT problem
    nx = ddpModel.dynamicModel.nx()
    nu = ddpModel.dynamicModel.nu()
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
    g[:nx] = ddpData.interval[0].x
    w_i = np.zeros((nw,1))
    q_i = np.zeros((nw,1))
    Q_i = np.zeros((nw,nw))
    f_i = np.zeros((2*nx,nw))
    for k in xrange(N):
      # Interval data
      data = ddpData.interval[k]

      # State, control and decision vector
      x_i = data.x
      u_i = data.u
      w_i[:nx] = x_i
      w_i[nx:] = u_i

      # Running cost and its derivatives
      lx_i = data.cost.lx
      lu_i = data.cost.lu
      lxx_i = data.cost.lxx
      luu_i = data.cost.luu
      lux_i = data.cost.lux
      q_i[:nx] = lx_i
      q_i[nx:] = lu_i
      Q_i[:nx,:nx] = lxx_i
      Q_i[:nx,nx:] = lux_i.T
      Q_i[nx:,:nx] = lux_i
      Q_i[nx:,nx:] = luu_i

      # Dynamics and its derivatives
      fx_i = data.dynamic.discretizer.fx
      fu_i = data.dynamic.discretizer.fu
      f_i[:nx,:nx] = -Ixx
      f_i[nx:,:nx] = fx_i
      f_i[nx:,nx:] = fu_i

      # Updating the constraint and cost functions and their gradient, and the
      # Hessian of this problem
      hess[k*nw:(k+1)*nw, k*nw:(k+1)*nw] = Q_i
      jac[k*nx:(k+2)*nx, k*nw:(k+1)*nw] = f_i
      grad[k*nw:(k+1)*nw] = q_i
      g[k*nx:(k+2)*nx] += np.dot(f_i, w_i)

    # Terminal state and cost derivatives
    x_T = ddpData.interval[-1].x
    lx_T = ddpData.interval[-1].cost.lx
    lxx_T = ddpData.interval[-1].cost.lxx

    # Updating the terms regarding the terminal condition
    hess[N*nw:N*nw+nx, N*nw:N*nw+nx] = lxx_T
    jac[N*nx:(N+2)*nx, N*nw:N*nw+nx] = -Ixx
    grad[N*nw:(N+1)*nw] = lx_T
    g[N*nx:(N+2)*nx] += -x_T

    # Computing the KKT matrix and its vector
    kkt_mat = np.zeros((nvar+ncon,nvar+ncon))
    kkt_vec = np.zeros((nvar+ncon,1))
    kkt_mat[:nvar,:nvar] = hess
    kkt_mat[:nvar,nvar:] = jac.T
    kkt_mat[nvar:,:nvar] = jac
    kkt_vec[:nvar] = grad
    kkt_vec[nvar:] = g

    # Solving the KKT problem
    sol = np.linalg.solve(kkt_mat, -kkt_vec)
    w = sol[:nvar]
    lag = sol[nvar:]
    return w, lag, hess, jac, grad, g

if __name__ == '__main__':
  unittest.main()