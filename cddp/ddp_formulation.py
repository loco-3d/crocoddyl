import numpy as np
from numpy.linalg import inv


class DDPPhase:

  def __init__(self,model,running_cost,terminal_cost,integrator,timeline):
    self.model = model
    self.running_cost = running_cost
    self.terminal_cost = terminal_cost
    self.integrator = integrator
    self.timeline = timeline
    self.N = len(timeline)-1

    ## Allocation of data
    self.intervals = [DDPRunningInterval(model,running_cost) for k in range(self.N)]
    for k,it in enumerate(self.intervals):
      it.t0 = timeline[k]
      it.t1 = timeline[k+1]


    self.intervals.append(DDPFinalInterval(model,terminal_cost))
    self.terminal_interval = self.intervals[-1]
    self.intial_interval = self.intervals[0]

    self.total_cost = float('Inf')

    # Regularization

    self.mu1 = 0e-6
    self.mu2 = 0e-6

  def setInitalState(self,x0):
    self.intial_interval.x0[:] = x0

  def initControl(self,controls):
    assert len(controls) == self.N
    for k in range(self.N):
      it = self.intervals[k]
      it.u[:] = controls[k]

  def forwardPass(self,update_control=True):
    it = self.intial_interval
    x_new = it.x0.copy()
    for k in range(self.N):
      it = self.intervals[k]
      if update_control:
        it.u += it.K * (x_new - it.x0) + it.j
      dt = it.t1 - it.t0
      it.x0[:] = x_new
      x_next_new = self.integrator.integrate(self.model,it.model_data,it.t0,it.x0,it.u,dt)
      it.x1[:] = x_next_new
      x_new = x_next_new

    it_terminal = self.terminal_interval
    it_terminal.x[:] = x_next_new

  def evalObjectiveFunction(self):
    total_cost = 0.
    for it in self.intervals[:-1]:
      dt = it.t1 - it.t0
      total_cost += self.running_cost.l(it.x0,it.u,it.cost_data) * dt

    terminal_interval = self.terminal_interval
    total_cost += self.terminal_cost.l(terminal_interval.x,terminal_interval.cost_data)

    self.total_cost = total_cost
    return total_cost

  def backwardPass(self):

    # Terminal interval
    terminal_cost = self.terminal_cost
    terminal_interval = self.terminal_interval

    x_terminal = terminal_interval.x
    terminal_interval.V = terminal_cost.l(x_terminal,terminal_interval.cost_data) # V
    terminal_interval.A = terminal_cost.l_xx(x_terminal,terminal_interval.cost_data) # Vxx
    terminal_interval.b = terminal_cost.l_x(x_terminal,terminal_interval.cost_data) # Vx

    # Backward pass
    running_cost = self.running_cost
    model = self.model

    for k in range(self.N-1,-1,-1):
      it = self.intervals[k]
      it_next = self.intervals[k+1]

      dt = it.t1 - it.t0
      model_data = it.model_data
      cost_data = it.cost_data
      x = it.x0
      u = it.u
      model.f_x(x,u,model_data)
      f_x = model_data.m_fx * dt + np.eye(model.n)
      model.f_u(x, u, model_data)
      f_u = model_data.m_fu * dt

      A_reg = it_next.A.copy()
      A_reg[np.diag_indices_from(A_reg)] += self.mu1


      it.Q = running_cost.l(x,u,cost_data) * dt + it_next.V
      #print "l(x,u)",running_cost.l(x,u,cost_data)
      #print "lu", running_cost.l_u(x, u, cost_data)
      #print "luu", running_cost.l_uu(x, u, cost_data)
      #print "k -",k,"Q:",it.Q

      it.Q_x = running_cost.l_x(x,u,cost_data) * dt + f_x.T * it_next.b
      it.Q_u = running_cost.l_u(x,u,cost_data) * dt + f_u.T * it_next.b
      it.Q_xx = running_cost.l_xx(x,u,cost_data) * dt + f_x.T * (A_reg * f_x)
      it.Q_uu = running_cost.l_uu(x,u,cost_data) * dt + f_u.T * (A_reg * f_u)
      it.Q_xu = running_cost.l_xu(x,u,cost_data) * dt + f_x.T * (A_reg * f_u)

      #print "Quu",it.Q_uu
      #print "Qxu",it.Q_xu
      it.Q_uu[np.diag_indices_from(it.Q_uu)] += self.mu2

      Q_uu_inv = inv(it.Q_uu)
      K = -Q_uu_inv * it.Q_xu.T
      #print "K",K
      j = -Q_uu_inv * it.Q_u
      #print "j",j

      it.K = K
      it.j = j

      A = it.Q_xx.copy()
      A += K.T * it.Q_uu * K
      A += it.Q_xu * K
      A += K.T * it.Q_xu.T

      it.A = A
      #print "A",A

      b = it.Q_x.copy()
      #b -= K.T * (it.Q_uu * j)
      b += K.T * (it.Q_uu * j)
      b += it.Q_xu * j
      b += K.T * it.Q_u

      it.b = b

      it.c = -0.5*it.j.T*(it.Q_uu*it.j)

  def getControlTrajectory(self):

    control_traj = np.matrix(np.zeros((self.model.m,self.N)))
    for k,it in enumerate(self.intervals[:-1]):
      control_traj[:,k] = it.u

    return control_traj

  def getStateTrajectory(self):

    state_traj = np.matrix(np.zeros((self.model.n,self.N+1)))
    for k,it in enumerate(self.intervals[:-1]):
      state_traj[:,k] = it.x0

    state_traj[:,-1] = self.terminal_interval.x

    return state_traj




class DDPRunningInterval:
  """
  Class containing the date related to one interval of the DDP
  """

  def __init__(self,model,cost):
    self.model_data = model.createData()
    self.cost_data = cost.createData(self.model_data)

    self.n = model.n
    self.m = model.m

    # control on the interval
    self.u = np.matrix(np.zeros((model.m,1)))

    # starting state in the interval
    self.x0 = np.matrix(np.zeros((model.n,1)))
    # final state on the interval
    self.x1 = np.matrix(np.zeros((model.n,1)))

    # start and terminal time on the interval
    self.t0 = 0.
    self.t1 = 0.

    # Value function
    self.V = float('Inf')
    self.Q = float('Inf')
    self.Q_x = np.matrix(np.zeros((self.n,1))); self.Q_x.fill(float('Inf'))
    self.Q_u = np.matrix(np.zeros((self.m,1))); self.Q_u.fill(float('Inf'))
    self.Q_xx = np.matrix(np.zeros((self.n,self.n))); self.Q_xx.fill(float('Inf'))
    self.Q_xu = np.matrix(np.zeros((self.n,self.m))); self.Q_xu.fill(float('Inf'))
    self.Q_uu = np.matrix(np.zeros((self.m,self.m))); self.Q_uu.fill(float('Inf'))

    # Feedback
    self.K = np.matrix(np.zeros((self.m,self.n))); self.K.fill(float('Inf'))
    self.j = np.matrix(np.zeros((self.m,1))); self.j.fill(float('Inf'))

    # Quadratic approximation of the value function
    self.A = np.matrix(np.zeros((self.n,self.n))); self.A.fill(float('Inf'))
    self.b = np.matrix(np.zeros((self.n,1))); self.b.fill(float('Inf'))
    self.c = float('Inf')


class DDPFinalInterval:
  """
  Class containing the date related to one interval of the DDP
  """

  def __init__(self,model,cost):
    self.model_data = model.createData()
    self.cost_data = cost.createData(self.model_data)

    self.n = model.n

    # state of the final interval
    self.x = np.matrix(np.zeros((model.n,1)))

    # time of the final interval
    self.t = 0.

    # Value function
    self.V = float('Inf')

    # Quadratic approximation of the value function
    self.A = np.matrix(np.zeros((self.n,self.n))); self.A.fill(float('Inf'))
    self.b = np.matrix(np.zeros((self.n,1))); self.b.fill(float('Inf'))
    self.c = float('Inf')
