from .libcrocoddyl_pywrap import *
from .libcrocoddyl_pywrap import __version__


def setGepettoViewerBackground(robot):
    if not hasattr(robot, 'viewer'):
        # Spawn robot model
        robot.initViewer(loadModel=True)
        # Set white background and floor
        window_id = robot.viewer.gui.getWindowID('python-pinocchio')
        robot.viewer.gui.setBackgroundColor1(window_id, [1., 1., 1., 1.])
        robot.viewer.gui.setBackgroundColor2(window_id, [1., 1., 1., 1.])
        robot.viewer.gui.addFloor('hpp-gui/floor')
        robot.viewer.gui.setScale('hpp-gui/floor', [0.5, 0.5, 0.5])
        robot.viewer.gui.setColor('hpp-gui/floor', [0.7, 0.7, 0.7, 1.])
        robot.viewer.gui.setLightingMode('hpp-gui/floor', 'OFF')


def displayTrajectory(robot, xs, dt=0.1, rate=-1, cameraTF=None):
    """  Display a robot trajectory xs using Gepetto-viewer gui.

    :param robot: Robot wrapper
    :param xs: state trajectory
    :param dt: step duration
    :param rate: visualization rate
    :param cameraTF: camera transform
    """
    setGepettoViewerBackground(robot)
    if cameraTF is not None:
        robot.viewer.gui.setCameraTransform(0, cameraTF)
    import numpy as np

    import time
    S = 1 if rate <= 0 else max(len(xs) / rate, 1)
    for i, x in enumerate(xs):
        if not i % S:
            robot.display(x[:robot.nq])
            time.sleep(dt)


class CallbackDisplay(libcrocoddyl_pywrap.CallbackAbstract):
    def __init__(self, robotwrapper, rate=-1, freq=1, cameraTF=None):
        libcrocoddyl_pywrap.CallbackAbstract.__init__(self)
        self.robotwrapper = robotwrapper
        self.rate = rate
        self.cameraTF = cameraTF
        self.freq = freq

    def __call__(self, solver):
        if (solver.iter + 1) % self.freq:
            return
        dt = solver.models()[0].dt
        displayTrajectory(self.robotwrapper, solver.xs, dt, self.rate, self.cameraTF)


class CallbackLogger(libcrocoddyl_pywrap.CallbackAbstract):
    def __init__(self):
        libcrocoddyl_pywrap.CallbackAbstract.__init__(self)
        self.steps = []
        self.iters = []
        self.costs = []
        self.control_regs = []
        self.state_regs = []
        self.th_stops = []
        self.gm_stops = []
        self.xs = []
        self.us = []
        self.gaps = []

    def __call__(self, solver):
        import copy
        import numpy as np
        self.xs = copy.copy(solver.xs)
        self.us = copy.copy(solver.us)
        self.steps.append(solver.stepLength)
        self.iters.append(solver.iter)
        self.costs.append(solver.cost)
        self.control_regs.append(solver.u_reg)
        self.state_regs.append(solver.x_reg)
        self.th_stops.append(solver.stoppingCriteria())
        self.gm_stops.append(-np.asscalar(solver.expectedImprovement()[1]))
        self.gaps.append(copy.copy(solver.gaps))


def plotOCSolution(xs, us, figIndex=1, show=True):
    import matplotlib.pyplot as plt
    import numpy as np
    # Getting the state and control trajectories
    nx, nu = xs[0].shape[0], us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) for u in us]

    plt.figure(figIndex)

    # Plotting the state trajectories
    plt.subplot(211)
    [plt.plot(X[i], label='x' + str(i)) for i in range(nx)]
    plt.legend()

    # Plotting the control commands
    plt.subplot(212)
    [plt.plot(U[i], label='u' + str(i)) for i in range(nu)]
    plt.legend()
    plt.xlabel('knots')
    if show:
        plt.show()


def plotSolverConvergence(costs, muLM, muV, gamma, theta, alpha, figIndex=1, show=True):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figIndex, figsize=(6.4, 8))
    # Plotting the total cost sequence
    plt.subplot(511)
    plt.ylabel('cost')
    plt.plot(costs)

    # Ploting mu sequences
    plt.subplot(512)
    plt.ylabel('mu')
    plt.plot(muLM, label='LM')
    plt.plot(muV, label='V')
    plt.legend()

    # Plotting the gradient sequence (gamma and theta)
    plt.subplot(513)
    plt.ylabel('gamma')
    plt.plot(gamma)
    plt.subplot(514)
    plt.ylabel('theta')
    plt.plot(theta)

    # Plotting the alpha sequence
    plt.subplot(515)
    plt.ylabel('alpha')
    ind = np.arange(len(alpha))
    plt.bar(ind, alpha)
    plt.xlabel('iteration')
    if show:
        plt.show()
