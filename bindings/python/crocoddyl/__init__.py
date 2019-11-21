from .libcrocoddyl_pywrap import *
from .libcrocoddyl_pywrap import __version__


def rotationMatrixFromTwoVectors(a, b):
    import pinocchio
    import numpy as np

    a_copy = a / np.linalg.norm(a)
    b_copy = b / np.linalg.norm(b)
    a_cross_b = pinocchio.utils.cross(a_copy, b_copy)
    s = np.linalg.norm(a_cross_b)
    c = np.asscalar(a_copy.T * b_copy)
    ab_skew = pinocchio.utils.skew(a_cross_b)
    return np.matrix(np.eye(3)) + ab_skew + ab_skew * ab_skew * (1 - c) / s**2


class GepettoDisplay:
    def __init__(self, robot, rate=-1, freq=1, cameraTF=None, floor=True):
        self.robot = robot
        self.rate = rate
        self.freq = freq
        self.cameraTF = cameraTF
        self.setBackground(floor)
        if cameraTF is not None:
            robot.viewer.gui.setCameraTransform(0, cameraTF)

    def display(self, xs, fs=None, dt=0.1):
        import numpy as np
        if fs is not None:
            import pinocchio

            totalWeight = sum(m.mass for m in self.robot.model.inertias) * np.linalg.norm(self.robot.model.gravity.linear)
            forceGroup = "world/robot/contact_forces"
            forceRadius = 0.015
            forceLength = 0.5
            forceColor = [1., 0., 1., 1.]
            self.robot.viewer.gui.createGroup(forceGroup)
            for f in fs[0]:
                key = f['key']
                t, R = f['oMf'].translation, rotationMatrixFromTwoVectors(np.matrix([1., 0., 0.]).T, f['f'].linear)
                self.robot.viewer.gui.addArrow(forceGroup + "/" + key, forceRadius, forceLength, forceColor)

        import time
        S = 1 if self.rate <= 0 else max(len(xs) / self.rate, 1)
        for i, x in enumerate(xs):
            if not i % S:
                if fs is not None:
                    self.robot.viewer.gui.setFloatProperty(forceGroup, 'Alpha', 0.)
                    for f in fs[i]:
                        key = f['key']
                        self.robot.viewer.gui.setFloatProperty(forceGroup + "/" + key, 'Alpha', 1.)
                    for f in fs[i]:
                        key = f['key']
                        force = f['f'].linear
                        t, R = f['oMf'].translation, rotationMatrixFromTwoVectors(np.matrix([1., 0., 0.]).T, force)
                        pose = pinocchio.se3ToXYZQUATtuple(pinocchio.SE3(R, t))
                        forceMagnitud = np.linalg.norm(force) / totalWeight
                        self.robot.viewer.gui.setVector3Property(forceGroup + "/" + key, 'Scale', [1. * forceMagnitud, 1., 1.])
                        self.robot.viewer.gui.applyConfiguration(forceGroup + "/" + key, pose)
                self.robot.display(x[:self.robot.nq])
                time.sleep(dt)

    def setBackground(self, floor):
        if not hasattr(self.robot, 'viewer'):
            # Spawn robot model
            self.robot.initViewer(windowName="crocoddyl", loadModel=False)
            self.robot.loadViewerModel(rootNodeName="robot")
            # Set white background and floor
            window_id = self.robot.viewer.gui.getWindowID("crocoddyl")
            self.robot.viewer.gui.setBackgroundColor1(window_id, [1., 1., 1., 1.])
            self.robot.viewer.gui.setBackgroundColor2(window_id, [1., 1., 1., 1.])
            self.robot.viewer.gui.createGroup("world/floor")
            if floor:
                self.robot.viewer.gui.addFloor("world/floor/flat")
                self.robot.viewer.gui.setScale("world/floor/flat", [0.5, 0.5, 0.5])
                self.robot.viewer.gui.setColor("world/floor/flat", [0.7, 0.7, 0.7, 1.])
                self.robot.viewer.gui.setLightingMode("world/floor/flat", 'OFF')



class CallbackDisplay(libcrocoddyl_pywrap.CallbackAbstract):
    def __init__(self, display):
        libcrocoddyl_pywrap.CallbackAbstract.__init__(self)
        self.visualization = display

    def __call__(self, solver):
        if (solver.iter + 1) % self.visualization.freq:
            return
        dt = solver.models()[0].dt

        fs = []
        for data in solver.datas():
            if hasattr(data, "differential"):
                if isinstance(data.differential, libcrocoddyl_pywrap.DifferentialActionDataContactFwdDynamics):
                    fc = []
                    for key, contact in data.differential.contacts.contacts.items():
                        oMf = contact.pinocchio.oMi[contact.joint] * contact.jMf
                        force = contact.jMf.actInv(contact.f)
                        fc.append({"key": str(contact.joint), "oMf": oMf, "f": force})
                    fs.append(fc)
            elif isinstance(data, libcrocoddyl_pywrap.ActionDataImpulseFwdDynamics):
                fc = []
                for key, impulse in data.impulses.impulses.items():
                    force = impulse.jMf.actInv(impulse.f)
                    oMf = impulse.pinocchio.oMi[impulse.joint] * impulse.jMf
                    fc.append({"key": str(impulse.joint), "oMf": oMf, "f": force})
                fs.append(fc)
        self.visualization.display(solver.xs, fs, dt)


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


def plotOCSolution(xs=None, us=None, figIndex=1, show=True):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Getting the state and control trajectories
    if xs is not None:
        xsPlotIdx = 111
        nx = xs[0].shape[0]
        X = [0.] * nx
        for i in range(nx):
            X[i] = [np.asscalar(x[i]) for x in xs]
    if us is not None:
        usPlotIdx = 111
        nu = us[0].shape[0]
        U = [0.] * nu
        for i in range(nu):
            U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
    if xs is not None and us is not None:
        xsPlotIdx = 211
        usPlotIdx = 212
    plt.figure(figIndex)

    # Plotting the state trajectories
    if xs is not None:
        plt.subplot(xsPlotIdx)
        [plt.plot(X[i], label='x' + str(i)) for i in range(nx)]
        plt.legend()

    # Plotting the control commands
    if us is not None:
        plt.subplot(usPlotIdx)
        [plt.plot(U[i], label='u' + str(i)) for i in range(nu)]
        plt.legend()
        plt.xlabel('knots')
    if show:
        plt.show()


def plotConvergence(costs, muLM, muV, gamma, theta, alpha, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figIndex, figsize=(6.4, 8))
    plt.suptitle(figTitle, fontsize=14)

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
