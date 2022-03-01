from .libcrocoddyl_pywrap import *
from .libcrocoddyl_pywrap import __version__
from .deprecation import *

import pinocchio
import numpy as np
import time
import warnings


def rotationMatrixFromTwoVectors(a, b):
    a_copy = a / np.linalg.norm(a)
    b_copy = b / np.linalg.norm(b)
    a_cross_b = np.cross(a_copy, b_copy, axis=0)
    s = np.linalg.norm(a_cross_b)
    if libcrocoddyl_pywrap.getNumpyType() == np.matrix:
        warnings.warn("Numpy matrix supports will be removed in future release", DeprecationWarning, stacklevel=2)
        if s == 0:
            return np.matrix(np.eye(3))
        c = np.asscalar(a_copy.T * b_copy)
        ab_skew = pinocchio.skew(a_cross_b)
        return np.matrix(np.eye(3)) + ab_skew + ab_skew * ab_skew * (1 - c) / s**2
    else:
        if s == 0:
            return np.eye(3)
        c = np.dot(a_copy, b_copy)
        ab_skew = pinocchio.skew(a_cross_b)
        return np.eye(3) + ab_skew + np.dot(ab_skew, ab_skew) * (1 - c) / s**2


class DisplayAbstract:

    def __init__(self, rate=-1, freq=1):
        self.rate = rate
        self.freq = freq

    def displayFromSolver(self, solver, factor=1.):
        numpy_conversion = False
        if libcrocoddyl_pywrap.getNumpyType() == np.matrix:
            numpy_conversion = True
            libcrocoddyl_pywrap.switchToNumpyMatrix()
            warnings.warn("Numpy matrix supports will be removed in future release", DeprecationWarning, stacklevel=2)
        fs = self.getForceTrajectoryFromSolver(solver)
        ps = self.getFrameTrajectoryFromSolver(solver)

        models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
        dts = [m.dt if hasattr(m, "differential") else 0. for m in models]
        self.display(solver.xs, fs, ps, dts, factor)
        if numpy_conversion:
            numpy_conversion = False
            libcrocoddyl_pywrap.switchToNumpyMatrix()

    def display(self, xs, fs=[], ps=[], dts=[], factor=1.):
        """ Display the state, force and frame trajectories"""
        raise NotImplementedError("Not implemented yet.")

    def getForceTrajectoryFromSolver(self, solver):
        """ Get the force trajectory from the solver"""
        return None

    def getFrameTrajectoryFromSolver(self, solver):
        """ Get the frame trajectory from the solver"""
        return None


class GepettoDisplay(DisplayAbstract):

    def __init__(self, robot, rate=-1, freq=1, cameraTF=None, floor=True, frameNames=[], visibility=False):
        DisplayAbstract.__init__(self, rate, freq)
        self.robot = robot

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

        self._addRobot()
        self._setBackground()
        if cameraTF is not None:
            self.robot.viewer.gui.setCameraTransform(self.robot.viz.windowID, cameraTF)
        if floor:
            self._addFloor()
        self.totalWeight = sum(m.mass
                               for m in self.robot.model.inertias) * np.linalg.norm(self.robot.model.gravity.linear)
        self.x_axis = np.array([1., 0., 0.])
        self.z_axis = np.array([0., 0., 1.])
        self.robot.viewer.gui.createGroup(self.forceGroup)
        self.robot.viewer.gui.createGroup(self.frictionGroup)
        self.robot.viewer.gui.createGroup(self.frameTrajGroup)
        self._addForceArrows()
        self._addFrameCurves()
        self._addFrictionCones()

    def display(self, xs, fs=[], ps=[], dts=[], factor=1.):
        numpy_conversion = False
        if libcrocoddyl_pywrap.getNumpyType() == np.matrix:
            numpy_conversion = True
            libcrocoddyl_pywrap.switchToNumpyMatrix()
            warnings.warn("Numpy matrix supports will be removed in future release", DeprecationWarning, stacklevel=2)
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
                        self.robot.viewer.gui.applyConfiguration(forceName, forcePose)
                        self.robot.viewer.gui.setVector3Property(forceName, "Scale", [1. * forceMagnitud, 1., 1.])
                        self.robot.viewer.gui.setVisibility(forceName, "ON")
                        # Display the friction cones
                        position = pose
                        position.rotation = f["R"]
                        frictionName = self.frictionGroup + "/" + key
                        self._setConeMu(key, f["mu"])
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
        if numpy_conversion:
            numpy_conversion = False
            libcrocoddyl_pywrap.switchToNumpyMatrix()

    def getForceTrajectoryFromSolver(self, solver):
        if len(self.frameTrajNames) == 0:
            return None
        fs = []
        models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
        datas = solver.problem.runningDatas.tolist() + [solver.problem.terminalData]
        for i, data in enumerate(datas):
            model = models[i]
            if hasattr(data, "differential"):
                if isinstance(data.differential, libcrocoddyl_pywrap.DifferentialActionDataContactFwdDynamics):
                    fc = []
                    for key, contact in data.differential.multibody.contacts.contacts.todict().items():
                        if model.differential.contacts.contacts[key].active:
                            joint = model.differential.state.pinocchio.frames[contact.frame].parent
                            oMf = contact.pinocchio.oMi[joint] * contact.jMf
                            fiMo = pinocchio.SE3(contact.pinocchio.oMi[joint].rotation.T, contact.jMf.translation)
                            force = fiMo.actInv(contact.f)
                            R = np.eye(3)
                            mu = 0.7
                            for k, c in model.differential.costs.costs.todict().items():
                                if isinstance(c.cost.residual, libcrocoddyl_pywrap.ResidualModelContactFrictionCone):
                                    if contact.frame == c.cost.residual.id:
                                        R = c.cost.residual.reference.R
                                        mu = c.cost.residual.reference.mu
                                        continue
                            fc.append({"key": str(joint), "oMf": oMf, "f": force, "R": R, "mu": mu})
                    fs.append(fc)
                elif isinstance(data.differential, libcrocoddyl_pywrap.StdVec_DiffActionData):
                    fc = []
                    for key, contact in data.differential[0].multibody.contacts.contacts.todict().items():
                        if model.differential.contacts.contacts[key].active:
                            joint = model.differential.state.pinocchio.frames[contact.frame].parent
                            oMf = contact.pinocchio.oMi[joint] * contact.jMf
                            fiMo = pinocchio.SE3(contact.pinocchio.oMi[joint].rotation.T, contact.jMf.translation)
                            force = fiMo.actInv(contact.f)
                            R = np.eye(3)
                            mu = 0.7
                            for k, c in model.differential.costs.costs.todict().items():
                                if isinstance(c.cost.residual, libcrocoddyl_pywrap.ResidualModelContactFrictionCone):
                                    if contact.frame == c.cost.residual.id:
                                        R = c.cost.residual.reference.R
                                        mu = c.cost.residual.reference.mu
                                        continue
                            fc.append({"key": str(joint), "oMf": oMf, "f": force, "R": R, "mu": mu})
                    fs.append(fc)
            elif isinstance(data, libcrocoddyl_pywrap.ActionDataImpulseFwdDynamics):
                fc = []
                for key, impulse in data.multibody.impulses.impulses.todict().items():
                    if model.impulses.impulses[key].active:
                        joint = model.state.pinocchio.frames[impulse.frame].parent
                        oMf = impulse.pinocchio.oMi[joint] * impulse.jMf
                        fiMo = pinocchio.SE3(impulse.pinocchio.oMi[joint].rotation.T, impulse.jMf.translation)
                        force = fiMo.actInv(impulse.f)
                        R = np.eye(3)
                        mu = 0.7
                        for k, c in model.costs.costs.todict().items():
                            if isinstance(c.cost.residual, libcrocoddyl_pywrap.ResidualModelContactFrictionCone):
                                if impulse.frame == c.cost.residual.id:
                                    R = c.cost.residual.reference.R
                                    mu = c.cost.residual.reference.mu
                                    continue
                        fc.append({"key": str(joint), "oMf": oMf, "f": force, "R": R, "mu": mu})
                fs.append(fc)
        return fs

    def getFrameTrajectoryFromSolver(self, solver):
        if len(self.frameTrajNames) == 0:
            return None
        ps = {fr: [] for fr in self.frameTrajNames}
        models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
        datas = solver.problem.runningDatas.tolist() + [solver.problem.terminalData]
        for key, p in ps.items():
            frameId = int(key)
            for i, data in enumerate(datas):
                model = models[i]
                if hasattr(data, "differential"):
                    # Update the frame placement if there is not contact.
                    # Note that, in non-contact cases, the action model does not compute it for efficiency reason
                    if isinstance(data.differential, libcrocoddyl_pywrap.StdVec_DiffActionData):
                        differential = data.differential[0]
                    else:
                        differential = data.differential
                    if len(differential.multibody.contacts.contacts.todict().items()) == 0:
                        pinocchio.updateFramePlacement(model.differential.state.pinocchio,
                                                       differential.multibody.pinocchio, frameId)
                    pose = differential.multibody.pinocchio.oMf[frameId]
                    p.append(np.asarray(pose.translation.T).reshape(-1).tolist())
                elif isinstance(data, libcrocoddyl_pywrap.ActionDataImpulseFwdDynamics):
                    pose = data.multibody.pinocchio.oMf[frameId]
                    p.append(np.asarray(pose.translation.T).reshape(-1).tolist())
        return ps

    def _addRobot(self):
        # Spawn robot model
        self.robot.initViewer(windowName="crocoddyl", loadModel=False)
        self.robot.loadViewerModel(rootNodeName="robot")

    def _setBackground(self):
        # Set white background and floor
        window_id = self.robot.viewer.gui.getWindowID("crocoddyl")
        self.robot.viewer.gui.setBackgroundColor1(window_id, self.backgroundColor)
        self.robot.viewer.gui.setBackgroundColor2(window_id, self.backgroundColor)

    def _addFloor(self):
        self.robot.viewer.gui.createGroup(self.floorGroup)
        self.robot.viewer.gui.addFloor(self.floorGroup + "/flat")
        self.robot.viewer.gui.setScale(self.floorGroup + "/flat", self.floorScale)
        self.robot.viewer.gui.setColor(self.floorGroup + "/flat", self.floorColor)
        self.robot.viewer.gui.setLightingMode(self.floorGroup + "/flat", "OFF")

    def _addForceArrows(self):
        for key in self.activeContacts:
            forceName = self.forceGroup + "/" + key
            self.robot.viewer.gui.addArrow(forceName, self.forceRadius, self.forceLength, self.forceColor)
            self.robot.viewer.gui.setFloatProperty(forceName, "Alpha", 1.)
        if self.fullVisibility:
            self.robot.viewer.gui.setVisibility(self.forceGroup, "ALWAYS_ON_TOP")

    def _addFrictionCones(self):
        for key in self.activeContacts:
            self._createCone(key, self.frictionConeScale, mu=0.7)

    def _addFrameCurves(self):
        for key in self.frameTrajNames:
            frameName = self.frameTrajGroup + "/" + key
            self.robot.viewer.gui.addCurve(frameName, [np.array([0., 0., 0.]).tolist()] * 2, self.frameTrajColor[key])
            self.robot.viewer.gui.setCurveLineWidth(frameName, self.frameTrajLineWidth)
            if self.fullVisibility:
                self.robot.viewer.gui.setVisibility(frameName, "ALWAYS_ON_TOP")

    def _createCone(self, coneName, scale=1., mu=0.7):
        m_generatrices = np.matrix(np.empty([3, 4]))
        m_generatrices[:, 0] = np.matrix([mu, mu, 1.]).T
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

    def _setConeMu(self, coneName, mu):
        current_mu = self.frictionMu[coneName]
        if mu != current_mu:
            self.frictionMu[coneName] = mu
            coneGroup = self.frictionGroup + "/" + coneName

            self.robot.viewer.gui.deleteNode(coneGroup, True)
            self._createCone(coneName, self.frictionConeScale, mu)


class MeshcatDisplay(DisplayAbstract):

    def __init__(self, robot, rate=-1, freq=1, openWindow=True):
        DisplayAbstract.__init__(self, rate, freq)
        self.robot = robot
        robot.setVisualizer(
            pinocchio.visualize.MeshcatVisualizer(model=self.robot.model,
                                                  collision_model=self.robot.collision_model,
                                                  visual_model=self.robot.visual_model))
        self._addRobot(openWindow)

    def display(self, xs, fs=[], ps=[], dts=[], factor=1.):
        if not dts:
            dts = [0.] * len(xs)

        S = 1 if self.rate <= 0 else max(len(xs) // self.rate, 1)
        for i, x in enumerate(xs):
            if not i % S:
                self.robot.display(x[:self.robot.nq])
                time.sleep(dts[i] * factor)

    def _addRobot(self, openWindow):
        # Spawn robot model
        self.robot.initViewer(open=openWindow)
        self.robot.loadViewerModel(rootNodeName="robot")


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
        self.grads.append(-solver.expectedImprovement()[1].item())


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
