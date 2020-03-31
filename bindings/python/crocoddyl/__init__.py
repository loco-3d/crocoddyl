from .libcrocoddyl_pywrap import *
from .libcrocoddyl_pywrap import __version__

import pinocchio
import numpy as np
import time


def rotationMatrixFromTwoVectors(a, b):
    a_copy = a / np.linalg.norm(a)
    b_copy = b / np.linalg.norm(b)
    a_cross_b = np.cross(a_copy, b_copy, axis=0)
    s = np.linalg.norm(a_cross_b)
    if s == 0:
        return np.matrix(np.eye(3))
    c = np.asscalar(a_copy.T * b_copy)
    ab_skew = pinocchio.skew(a_cross_b)
    return np.matrix(np.eye(3)) + ab_skew + ab_skew * ab_skew * (1 - c) / s**2


class GepettoDisplay:
    def __init__(self, robot, rate=-1, freq=1, cameraTF=None, floor=True, frameNames=[], visibility=False):
        self.robot = robot
        self.rate = rate
        self.freq = freq

        # Visuals properties
        self.fullVisibility = visibility
        self.floorGroup = "world/floor"
        self.forceGroup = "world/robot/contact_forces"
        self.frictionGroup = "world/robot/friction_cone"
        self.frameTrajGroup = "world/robot/frame_trajectory"
        self.backgroundColor = [1., 1., 1., 1.]
        self.floorScale = [0.5, 0.5, 0.5]
        self.floorColor = [0.7, 0.7, 0.7, 1.]
        self.forceRadius = 0.015
        self.forceLength = 0.5
        self.forceColor = [1., 0., 1., 1.]
        self.frictionConeScale = 0.2
        self.frictionConeRays = True
        self.frictionConeColor1 = [0., 0.4, 0.79, 0.5]
        self.frictionConeColor2 = [0., 0.4, 0.79, 0.5]
        self.activeContacts = {}
        self.frictionMu = {}
        for n in frameNames:
            parentId = robot.model.frames[robot.model.getFrameId(n)].parent
            self.activeContacts[str(parentId)] = True
            self.frictionMu[str(parentId)] = 0.7
        self.frameTrajNames = []
        for n in frameNames:
            self.frameTrajNames.append(str(robot.model.getFrameId(n)))
        self.frameTrajColor = {}
        self.frameTrajLineWidth = 10
        for fr in self.frameTrajNames:
            self.frameTrajColor[fr] = list(np.hstack([np.random.choice(range(256), size=3) / 256., 1.]))

        self.addRobot()
        self.setBackground()
        if cameraTF is not None:
            self.robot.viewer.gui.setCameraTransform(0, cameraTF)
        if floor:
            self.addFloor()
        self.totalWeight = sum(m.mass
                               for m in self.robot.model.inertias) * np.linalg.norm(self.robot.model.gravity.linear)
        self.x_axis = np.matrix([1., 0., 0.]).T
        self.z_axis = np.matrix([0., 0., 1.]).T
        self.robot.viewer.gui.createGroup(self.forceGroup)
        self.robot.viewer.gui.createGroup(self.frictionGroup)
        self.robot.viewer.gui.createGroup(self.frameTrajGroup)
        self.addForceArrows()
        self.addFrameCurves()
        self.addFrictionCones()

    def display(self, xs, fs=[], ps=[], dts=[], factor=1.):
        if ps:
            for key, p in ps.items():
                self.robot.viewer.gui.setCurvePoints(self.frameTrajGroup + "/" + key, p)
        if not dts:
            dts = [0.] * len(xs)

        S = 1 if self.rate <= 0 else max(len(xs) / self.rate, 1)
        for i, x in enumerate(xs):
            if not i % S:
                if fs:
                    self.activeContacts = {k: False for k, c in self.activeContacts.items()}
                    for f in fs[i]:
                        key = f["key"]
                        pose = f["oMf"]
                        wrench = f["f"]
                        # Display the contact forces
                        R = rotationMatrixFromTwoVectors(self.x_axis, wrench.linear)
                        forcePose = pinocchio.SE3ToXYZQUATtuple(pinocchio.SE3(R, pose.translation))
                        forceMagnitud = np.linalg.norm(wrench.linear) / self.totalWeight
                        forceName = self.forceGroup + "/" + key
                        self.robot.viewer.gui.setVector3Property(forceName, "Scale", [1. * forceMagnitud, 1., 1.])
                        self.robot.viewer.gui.applyConfiguration(forceName, forcePose)
                        self.robot.viewer.gui.setVisibility(forceName, "ON")
                        # Display the friction cones
                        position = pose
                        position.rotation = rotationMatrixFromTwoVectors(self.z_axis, f["nsurf"])
                        frictionName = self.frictionGroup + "/" + key
                        self.setConeMu(key, f["mu"])
                        self.robot.viewer.gui.applyConfiguration(
                            frictionName, list(np.array(pinocchio.SE3ToXYZQUAT(position)).squeeze()))
                        self.robot.viewer.gui.setVisibility(frictionName, "ON")
                        self.activeContacts[key] = True
                for key, c in self.activeContacts.items():
                    if c == False:
                        self.robot.viewer.gui.setVisibility(self.forceGroup + "/" + key, "OFF")
                        self.robot.viewer.gui.setVisibility(self.frictionGroup + "/" + key, "OFF")
                self.robot.display(x[:self.robot.nq])
                time.sleep(dts[i] * factor)

    def displayFromSolver(self, solver, factor=1.):
        fs = self.getForceTrajectoryFromSolver(solver)
        ps = self.getFrameTrajectoryFromSolver(solver)

        models = solver.problem.runningModels + [solver.problem.terminalModel]
        dts = [m.dt if hasattr(m, "differential") else 0. for m in models]
        self.display(solver.xs, fs, ps, dts, factor)

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

    def addForceArrows(self):
        for key in self.activeContacts:
            forceName = self.forceGroup + "/" + key
            self.robot.viewer.gui.addArrow(forceName, self.forceRadius, self.forceLength, self.forceColor)
            self.robot.viewer.gui.setFloatProperty(forceName, "Alpha", 1.)
        if self.fullVisibility:
            self.robot.viewer.gui.setVisibility(self.forceGroup, "ALWAYS_ON_TOP")

    def addFrictionCones(self):
        for key in self.activeContacts:
            self.createCone(key, self.frictionConeScale, mu=0.7)

    def addFrameCurves(self):
        for key in self.frameTrajNames:
            frameName = self.frameTrajGroup + "/" + key
            self.robot.viewer.gui.addCurve(frameName, [np.array([0., 0., 0.]).tolist()] * 2, self.frameTrajColor[key])
            self.robot.viewer.gui.setCurveLineWidth(frameName, self.frameTrajLineWidth)
            if self.fullVisibility:
                self.robot.viewer.gui.setVisibility(frameName, "ALWAYS_ON_TOP")

    def getForceTrajectoryFromSolver(self, solver):
        fs = []
        models = solver.problem.runningModels + [solver.problem.terminalModel]
        datas = solver.problem.runningDatas + [solver.problem.terminalData]
        for i, data in enumerate(datas):
            model = models[i]
            if hasattr(data, "differential"):
                if isinstance(data.differential, libcrocoddyl_pywrap.DifferentialActionDataContactFwdDynamics):
                    fc = []
                    for key, contact in data.differential.multibody.contacts.contacts.items():
                        oMf = contact.pinocchio.oMi[contact.joint] * contact.jMf
                        force = contact.jMf.actInv(contact.f)
                        nsurf = np.matrix([0., 0., 1.]).T
                        mu = 0.7
                        for k, c in model.differential.costs.costs.items():
                            if isinstance(c.cost, libcrocoddyl_pywrap.CostModelContactFrictionCone):
                                if contact.joint == self.robot.model.frames[c.cost.reference.frame].parent:
                                    nsurf = c.cost.reference.oRf.nsurf
                                    mu = c.cost.reference.oRf.mu
                                    continue
                        fc.append({"key": str(contact.joint), "oMf": oMf, "f": force, "nsurf": nsurf, "mu": mu})
                    fs.append(fc)
            elif isinstance(data, libcrocoddyl_pywrap.ActionDataImpulseFwdDynamics):
                fc = []
                for key, impulse in data.multibody.impulses.impulses.items():
                    oMf = impulse.pinocchio.oMi[impulse.joint] * impulse.jMf
                    force = impulse.jMf.actInv(impulse.f)
                    nsurf = np.matrix([0., 0., 1.]).T
                    mu = 0.7
                    for k, c in model.costs.costs.items():
                        if isinstance(c.cost, libcrocoddyl_pywrap.CostModelContactFrictionCone):
                            if impulse.joint == self.robot.model.frames[c.cost.frame].parent:
                                nsurf = c.cost.friction_cone.nsurf
                                mu = c.cost.friction_cone.mu
                                continue
                    fc.append({"key": str(impulse.joint), "oMf": oMf, "f": force, "nsurf": nsurf, "mu": mu})
                fs.append(fc)
        return fs

    def getFrameTrajectoryFromSolver(self, solver):
        ps = {fr: [] for fr in self.frameTrajNames}
        datas = solver.problem.runningDatas + [solver.problem.terminalData]
        for key, p in ps.items():
            frameId = int(key)
            for data in datas:
                if hasattr(data, "differential"):
                    if hasattr(data.differential, "pinocchio"):
                        pose = data.differential.pinocchio.oMf[frameId]
                        p.append(np.asarray(pose.translation.T).reshape(-1).tolist())
                elif isinstance(data, libcrocoddyl_pywrap.ActionDataImpulseFwdDynamics):
                    if hasattr(data, "pinocchio"):
                        pose = data.pinocchio.oMf[frameId]
                        p.append(np.asarray(pose.translation.T).reshape(-1).tolist())
        return ps

    def createCone(self, coneName, scale=1., mu=0.7):
        m_generatrices = np.matrix(np.empty([3, 4]))
        m_generatrices[:, 0] = np.matrix([np.sqrt(2) / 2. * mu, np.sqrt(2) / 2. * mu, 1.]).T
        m_generatrices[:, 0] = m_generatrices[:, 0] / np.linalg.norm(m_generatrices[:, 0])
        m_generatrices[:, 1] = m_generatrices[:, 0]
        m_generatrices[0, 1] *= -1.
        m_generatrices[:, 2] = m_generatrices[:, 0]
        m_generatrices[:2, 2] *= -1.
        m_generatrices[:, 3] = m_generatrices[:, 0]
        m_generatrices[1, 3] *= -1.
        generatrices = m_generatrices
        v = [[0., 0., 0.]]
        for k in range(m_generatrices.shape[1]):
            v.append(m_generatrices[:3, k].T.tolist()[0])
        v.append(m_generatrices[:3, 0].T.tolist()[0])
        coneGroup = self.frictionGroup + "/" + coneName
        self.robot.viewer.gui.createGroup(coneGroup)

        meshGroup = coneGroup + "/cone"
        result = self.robot.viewer.gui.addCurve(meshGroup, v, self.frictionConeColor1)
        self.robot.viewer.gui.setCurveMode(meshGroup, 'TRIANGLE_FAN')
        if self.frictionConeRays:
            lineGroup = coneGroup + "/lines"
            self.robot.viewer.gui.createGroup(lineGroup)
            for k in range(m_generatrices.shape[1]):
                l = self.robot.viewer.gui.addLine(lineGroup + "/" + str(k), [0., 0., 0.],
                                                  m_generatrices[:3, k].T.tolist()[0], self.frictionConeColor2)
        self.robot.viewer.gui.setScale(coneGroup, [scale, scale, scale])
        if self.fullVisibility:
            self.robot.viewer.gui.setVisibility(meshGroup, "ALWAYS_ON_TOP")
            self.robot.viewer.gui.setVisibility(lineGroup, "ALWAYS_ON_TOP")

    def setConeMu(self, coneName, mu):
        current_mu = self.frictionMu[coneName]
        if mu != current_mu:
            self.frictionMu[coneName] = mu
            coneGroup = self.frictionGroup + "/" + coneName

            self.robot.viewer.gui.deleteNode(coneGroup + "/lines/0", "")
            self.robot.viewer.gui.deleteNode(coneGroup + "/lines/1", "")
            self.robot.viewer.gui.deleteNode(coneGroup + "/lines/2", "")
            self.robot.viewer.gui.deleteNode(coneGroup + "/lines/3", "")
            self.robot.viewer.gui.deleteNode(coneGroup + "/lines", "")
            self.robot.viewer.gui.deleteNode(coneGroup + "/cone", "")
            self.robot.viewer.gui.deleteNode(coneGroup, "")
            self.createCone(coneName, self.frictionConeScale, mu)


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
        self.xs = []
        self.us = []
        self.fs = []
        self.steps = []
        self.iters = []
        self.costs = []
        self.u_regs = []
        self.x_regs = []
        self.stops = []
        self.grads = []

    def __call__(self, solver):
        import copy
        self.xs = copy.copy(solver.xs)
        self.us = copy.copy(solver.us)
        self.fs.append(copy.copy(solver.fs))
        self.steps.append(solver.stepLength)
        self.iters.append(solver.iter)
        self.costs.append(solver.cost)
        self.u_regs.append(solver.u_reg)
        self.x_regs.append(solver.x_reg)
        self.stops.append(solver.stoppingCriteria())
        self.grads.append(-np.asscalar(solver.expectedImprovement()[1]))


def plotOCSolution(xs=None, us=None, figIndex=1, show=True, figTitle=""):
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
        plt.title(figTitle, fontsize=14)

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

    # Plotting the total cost sequence
    plt.subplot(511)
    plt.ylabel("cost")
    plt.plot(costs)
    plt.title(figTitle, fontsize=14)

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


def saveOCSolution(filename, xs, us, ks=None, Ks=None):
    import pickle
    data = {"xs": xs, "us": us, "ks": ks, "Ks": Ks}
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def saveConvergence(filename, costs, muLM, muV, gamma, theta, alpha):
    import pickle
    data = {"costs": costs, "muLM": muLM, "muV": muV, "gamma": gamma, "theta": theta, "alpha": alpha}
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def saveLogfile(filename, log):
    import pickle
    data = {
        "xs": log.xs,
        "us": log.us,
        "fs": log.fs,
        "steps": log.steps,
        "iters": log.iters,
        "costs": log.costs,
        "muLM": log.u_regs,
        "muV": log.x_regs,
        "stops": log.stops,
        "grads": log.grads
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
