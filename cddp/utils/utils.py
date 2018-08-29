import numpy as np


class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


def isPositiveDefinitive(A):
  """ Checks if the matrix is positive definitive.

  :param A matrix
  """
  try:
    _ = np.linalg.cholesky(A)
  except np.linalg.LinAlgError:
    return False
  return True

def assertClass(derivated, abstract):
  import inspect
  assert inspect.getmro(derivated.__class__)[-2].__name__ == abstract, \
    'The ' + derivated.__class__.__name__ + ' class has to derived from the ' +\
    abstract + ' abstract class.'

def assertGreaterThan(a, b):
  assert (a >= b).all(), 'The first vector is not greater than the second one.'

def visualizePlan(robot, x0, T, X, frame_idx=None, path=False):
  import gepetto.corbaserver
  cl = gepetto.corbaserver.Client()
  gui = cl.gui

  if gui.nodeExists("world"):
    gui.deleteNode("world",True)
  robot.initDisplay(loadModel=True)
  robot.display(x0[:robot.nq])

  gui.refresh()
  if frame_idx != None and path:
    ball_size = 0.04
    traj_node = "world/ee_ball"
    if gui.nodeExists(traj_node):
      gui.deleteNode(traj_node,True)
    gui.addSphere(traj_node, ball_size, [0., 1., 0., 1.])
  
  from time import sleep
  robot.initDisplay(loadModel=True)
  it = 0
  t0 = 0.
  for k in range(len(X)):
    qk = X[k][:robot.nq]
    robot.display(qk)
    if frame_idx != None and path:
      import pinocchio as se3
      M_pos = robot.framePosition(qk, frame_idx)
      gui.applyConfiguration(traj_node, se3.utils.se3ToXYZQUAT(M_pos))
      gui.addSphere(traj_node+str(it), ball_size/5, [0., 0., 1., 1.])
      gui.applyConfiguration(traj_node+str(it), se3.utils.se3ToXYZQUAT(M_pos))
      gui.refresh()
    sleep(T[k+1]-t0)
    t0 = T[k+1]
    it += 1

def plotDDPSolution(model, X, U, V):
  import matplotlib.pyplot as plt
  # Getting the joint position and commands
  q = []
  tau = []
  for i in range(model.nq):
    q.append([np.asscalar(k[i]) for k in X])
  for i in range(model.nv):
    tau.append([np.asscalar(k[i]) for k in U])

  plt.figure(1)

  # Plotting the joint position
  plt.subplot(311)
  [plt.plot(q[i], label='q'+str(i)) for i in range(model.nq)]
  plt.legend()
  plt.ylabel('rad')

  # Plotting the joint torques
  plt.subplot(312)
  [plt.plot(tau[i], label='u'+str(i)) for i in range(model.nv)]
  plt.legend()
  plt.ylabel('Nm')
  plt.xlabel('knots')

  # Plotting the total cost sequence
  plt.subplot(313)
  plt.plot(V)
  plt.xlabel('iteration')
  # plt.subplots_adjust(hspace=0.5)
  plt.show()


def plot(x, y, yerr=None, color=None, alpha_fill=0.3, ax=None):
  import matplotlib.pyplot as plt
  ax = ax if ax is not None else plt.gca()
  if color is None:
    color = ax._get_lines.color_cycle.next()
  ax.plot(x, y, color=color)
  if yerr is not None:
    if np.isscalar(yerr) or len(yerr) == len(y):
      ymin = y - yerr
      ymax = y + yerr
    elif len(yerr) == 2:
      ymin, ymax = yerr
    ax.fill_between(x,
      ymax.reshape(-1), ymin.reshape(-1), color=color, alpha=alpha_fill)


def show_plot():
  import matplotlib.pyplot as plt
  plt.show()