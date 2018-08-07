import numpy as np
from centroidal_dynamics_details import *
from integrator_base import *
from ddp_formulation import *
from time import time
import matplotlib.pyplot as plt

MASS = 55

model = CentroidalModelReduced(mass=MASS)
data = model.createData()
com0 = np.matrix([0.,0.,0.75]).T
com1 = com0 + np.matrix([1.,0.,0.]).T

def checkDerivativesModel(model,x,u):
  data = model.createData()
  data_ref = model.createData()

  n = model.n
  m = model.m
  eps = 1e-8

  # Check f_x
  model.f_x(x,u,data)
  model.f_u(x,u,data)

  model.f(x,u,data_ref)
  dx = data_ref.m_dx.copy()

  f_x_ref = np.matrix(np.zeros((n,n)))
  for k in range(n):
    x_fd = x.copy()
    x_fd[k] += eps
    model.f(x_fd, u, data_ref)
    f_x_ref[:,k] = (data_ref.m_dx - dx)/eps

  assert np.isclose(f_x_ref.A,data.m_fx.A).all()

  f_u_ref = np.matrix(np.zeros((n, m)))
  for k in range(m):
    u_fd = u.copy()
    u_fd[k] += eps
    model.f(x, u_fd, data_ref)
    f_u_ref[:,k] = (data_ref.m_dx - dx)/eps

  assert np.isclose(f_u_ref.A,data.m_fu.A).all()

def checkDerivativesRunningCost(model,cost,x,u):
  model_data = model.createData()
  model_data_fd = model.createData()

  cost_data = cost.createData(model_data)
  cost_data_fd = cost.createData(model_data_fd)
  n = model.n
  m = model.m
  eps = 1e-6

  # l_x
  cost.l_x(x,u,cost_data)
  l_x_ref = np.matrix(np.zeros([model.n,1]))

  l = cost.l(x,u,cost_data_fd)
  for k in range(n):
    x_fd = x.copy()
    x_fd[k] += eps

    l_x_fd = cost.l(x_fd,u,cost_data_fd)
    l_x_ref[k] = (l_x_fd - l)/eps

  print "l_x_ref\n",l_x_ref
  print "cost_data.m_lx\n",cost_data.m_lx
  assert np.isclose(l_x_ref.A, cost_data.m_lx.A,atol=2*eps).all()

  # l_xx
  cost.l_xx(x,u,cost_data)
  l_xx_ref = np.matrix(np.zeros([model.n,model.n]))

  l = cost.l(x,u,cost_data_fd)
  for k in range(n):
    x1 = x.copy()
    x1[k] += eps
    for i in range(n):
      x2 = x1.copy()
      x2[i] += eps

      l_xx_ref[k,i] = cost.l(x2, u, cost_data_fd)

      x2 = x1.copy()
      x2[i] -= eps
      l_xx_ref[k, i] -= cost.l(x2, u, cost_data_fd)

    x1 = x.copy()
    x1[k] -= eps
    for i in range(n):
      x2 = x1.copy()
      x2[i] += eps

      l_xx_ref[k, i] -= cost.l(x2, u, cost_data_fd)

      x2 = x1.copy()
      x2[i] -= eps
      l_xx_ref[k, i] += cost.l(x2, u, cost_data_fd)

  l_xx_ref /= (4*eps**2)
  print "l_xx_ref\n",l_xx_ref
  print "cost_data.m_lxx\n",cost_data.m_lxx

  assert np.isclose(l_xx_ref.A, cost_data.m_lxx.A,atol=2*eps).all()


  # l_u
  cost.l_u(x,u,cost_data)
  l_u_ref = np.matrix(np.zeros([model.m,1]))

  l = cost.l(x,u,cost_data_fd)
  for k in range(m):
    u_fd = u.copy()
    u_fd[k] += eps

    l_u_fd = cost.l(x,u_fd,cost_data_fd)
    l_u_ref[k] = (l_u_fd - l)/eps

  assert np.isclose(l_u_ref.A, cost_data.m_lu.A,atol=2*eps).all()

  # l_uu
  cost.l_uu(x,u,cost_data)
  l_uu_ref = np.matrix(np.zeros((model.m,model.m)))
  print l_uu_ref
  for k in range(m):
    u1 = u.copy()
    u1[k] += eps
    for i in range(m):
      u2 = u1.copy()
      u2[i] += eps

      l_uu_ref[k,i] = cost.l(x, u2, cost_data_fd)

      u2 = u1.copy()
      u2[i] -= eps
      l_uu_ref[k, i] -= cost.l(x, u2, cost_data_fd)

    u1 = u.copy()
    u1[k] -= eps
    for i in range(m):
      u2 = u1.copy()
      u2[i] += eps

      l_uu_ref[k, i] -= cost.l(x, u2, cost_data_fd)

      u2 = u1.copy()
      u2[i] -= eps
      l_uu_ref[k, i] += cost.l(x, u2, cost_data_fd)

  l_uu_ref /= (4*eps**2)


  assert np.isclose(l_uu_ref, cost_data.m_luu,atol=2*eps).all()




x0 = np.matrix(np.zeros((CentroidalModelReduced.n,1)))
x0[CentroidalModelReduced.com_position_id] = com0

x1 = np.matrix(np.zeros((CentroidalModelReduced.n,1)))
x1[CentroidalModelReduced.com_position_id] = com1

u = np.matrix(np.zeros((CentroidalModelReduced.m,1)))

dt = 0.05 # in s
t0 = 0.
tend = 2.
N = int(tend/dt)

timeline = np.linspace(t0,tend,N+1,endpoint=True)
initial_control = [np.matrix(np.zeros((model.m,1)))]*N

tic = time()
x_end = computeFlow(EulerIntegrator,model,data,timeline,x0,initial_control)
toc = time()
euler_ct = (toc - tic)*1e3

tic = time()
x_end = computeFlow(RK4Integrator,model,data,timeline,x0,initial_control)
toc = time()
rk4_ct = (toc - tic)*1e3

r1_cost = LeastControlCost(model)
r1_cost.setW(1e-3*np.eye(model.m))

r2_cost = LeastStateCost(model)
r2_cost.setTargetState(x1)

r_cost = SumOfRunningCost(model)
r_cost.append(r1_cost)
#r_cost.append(r2_cost)

t_cost = FinalStateCost(model)
t_cost.setFinalState(x1)


problem = DDPPhase(model,r_cost,t_cost,EulerIntegrator,timeline)
problem.setInitalState(x0)
problem.initControl(initial_control)

terminal_interval = problem.terminal_interval

problem.forwardPass(update_control=False)

K = 1
for k in range(K):

  problem.backwardPass()
  problem.forwardPass()

  cost_value = problem.evalObjectiveFunction()
  control_traj = problem.getControlTrajectory()
  print control_traj

  if not k%1:
    print "k -",k,"cost:",cost_value
    print "final state:",terminal_interval.x[0]
  #raw_input("Next")

control_traj = problem.getControlTrajectory()
state_traj = problem.getStateTrajectory()
timeline = problem.timeline

fig = plt.figure()
plt.subplot(211)
plt.plot(timeline,state_traj[0,:].A.squeeze(),marker="x")

plt.subplot(212)
plt.plot(timeline[:-1],control_traj[0,:].A.squeeze(),marker="x")

print 'ok'