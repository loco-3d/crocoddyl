from .libcrocoddyl_pywrap import *
from .libcrocoddyl_pywrap import __version__

import pinocchio
import numpy as np
import time


def rotationMatrixFromTwoVectors(a, b):
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

        # Visuals properties
        self.floorGroup = "world/floor"
        self.forceGroup = "world/robot/contact_forces"
        self.backgroundColor = [1., 1., 1., 1.]
        self.floorScale = [0.5, 0.5, 0.5]
        self.floorColor = [0.7, 0.7, 0.7, 1.]
        self.forceRadius = 0.015
        self.forceLength = 0.5
        self.forceColor = [1., 0., 1., 1.]

        self.addRobot()
        self.setBackground()
        if cameraTF is not None:
            self.robot.viewer.gui.setCameraTransform(0, cameraTF)

        # floor visuals properties
        if floor:
            self.addFloor()

        # Force visuals properties
        self.totalWeight = sum(m.mass
                               for m in self.robot.model.inertias) * np.linalg.norm(self.robot.model.gravity.linear)
        self.x_axis = np.matrix([1., 0., 0.]).T
        self.robot.viewer.gui.createGroup(self.forceGroup)

    def display(self, xs, fs=[], dts=[]):
        if fs:
            for f in fs[0]:
                key = f["key"]
                self.robot.viewer.gui.addArrow(self.forceGroup + "/" + key, self.forceRadius, self.forceLength,
                                               self.forceColor)
        if not dts:
            dts = [0.] * len(xs)

        S = 1 if self.rate <= 0 else max(len(xs) / self.rate, 1)
        for i, x in enumerate(xs):
            if not i % S:
                if fs:
                    self.robot.viewer.gui.setFloatProperty(self.forceGroup, "Alpha", 0.)
                    for f in fs[i]:
                        key = f["key"]
                        self.robot.viewer.gui.setFloatProperty(self.forceGroup + "/" + key, "Alpha", 1.)
                    for f in fs[i]:
                        key = f["key"]
                        force = f["f"].linear
                        t, R = f["oMf"].translation, rotationMatrixFromTwoVectors(self.x_axis, force)
                        pose = pinocchio.se3ToXYZQUATtuple(pinocchio.SE3(R, t))
                        forceMagnitud = np.linalg.norm(force) / self.totalWeight
                        self.robot.viewer.gui.setVector3Property(self.forceGroup + "/" + key, "Scale",
                                                                 [1. * forceMagnitud, 1., 1.])
                        self.robot.viewer.gui.applyConfiguration(self.forceGroup + "/" + key, pose)
                self.robot.display(x[:self.robot.nq])
                time.sleep(dts[i])

    def displayFromSolver(self, solver):
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

        dts = [m.dt if hasattr(m, "differential") else 0. for m in solver.models()]
        self.display(solver.xs, fs, dts)

    def addRobot(self):
        # Spawn robot model
        self.robot.initViewer(windowName="crocoddyl", loadModel=False)
        self.robot.loadViewerModel(rootNodeName="robot")

    def setBackground(self):
        # Set white background and floor
        window_id = self.robot.viewer.gui.getWindowID("crocoddyl")
        self.robot.viewer.gui.setBackgroundColor1(window_id, self.backgroundColor)
        self.robot.viewer.gui.setBackgroundColor2(window_id, self.backgroundColor)

    def addFloor(self):
        self.robot.viewer.gui.createGroup(self.floorGroup)
        self.robot.viewer.gui.addFloor(self.floorGroup + "/flat")
        self.robot.viewer.gui.setScale(self.floorGroup + "/flat", self.floorScale)
        self.robot.viewer.gui.setColor(self.floorGroup + "/flat", self.floorColor)
        self.robot.viewer.gui.setLightingMode(self.floorGroup + "/flat", "OFF")


class CallbackDisplay(libcrocoddyl_pywrap.CallbackAbstract):
    def __init__(self, display):
        libcrocoddyl_pywrap.CallbackAbstract.__init__(self)
        self.visualization = display

    def __call__(self, solver):
        if (solver.iter + 1) % self.visualization.freq:
            return
        self.visualization.displayFromSolver(solver)


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
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

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
        [plt.plot(X[i], label="x" + str(i)) for i in range(nx)]
        plt.legend()

    # Plotting the control commands
    if us is not None:
        plt.subplot(usPlotIdx)
        [plt.plot(U[i], label="u" + str(i)) for i in range(nu)]
        plt.legend()
        plt.xlabel("knots")
    if show:
        plt.show()


def plotConvergence(costs, muLM, muV, gamma, theta, alpha, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.figure(figIndex, figsize=(6.4, 8))
    plt.suptitle(figTitle, fontsize=14)

    # Plotting the total cost sequence
    plt.subplot(511)
    plt.ylabel("cost")
    plt.plot(costs)

    # Ploting mu sequences
    plt.subplot(512)
    plt.ylabel("mu")
    plt.plot(muLM, label="LM")
    plt.plot(muV, label="V")
    plt.legend()

    # Plotting the gradient sequence (gamma and theta)
    plt.subplot(513)
    plt.ylabel("gamma")
    plt.plot(gamma)
    plt.subplot(514)
    plt.ylabel("theta")
    plt.plot(theta)

    # Plotting the alpha sequence
    plt.subplot(515)
    plt.ylabel("alpha")
    ind = np.arange(len(alpha))
    plt.bar(ind, alpha)
    plt.xlabel("iteration")
    if show:
        plt.show()
