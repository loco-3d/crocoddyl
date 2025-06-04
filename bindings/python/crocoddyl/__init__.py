# ruff: noqa: F405
import inspect
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import pinocchio
from pinocchio.visualize import MeshcatVisualizer

from .libcrocoddyl_pywrap_float64 import *  # noqa: F403
from .libcrocoddyl_pywrap_float64 import __raw_version__, __version__  # noqa: F401


def rotationMatrixFromTwoVectors(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return np.eye(3)
    a_copy = a / a_norm
    b_copy = b / b_norm
    a_cross_b = np.cross(a_copy, b_copy, axis=0)
    s = np.linalg.norm(a_cross_b)
    if s == 0:
        return np.eye(3)
    c = np.dot(a_copy, b_copy)
    ab_skew = pinocchio.skew(a_cross_b)
    return np.eye(3) + ab_skew + np.dot(ab_skew, ab_skew) * (1 - c) / s**2


class DisplayAbstract(ABC):
    def __init__(self, robot, rate=-1, freq=1):
        self.robot = robot
        self._nv_root = 0.0
        self.rate = rate
        self.freq = freq
        # Visual properties
        self.frameTrajGroup = "world/robot/frame_trajectory"
        self.forceGroup = "world/robot/contact_forces"
        self.thrustGroup = "world/robot/thrust"
        self.frictionGroup = "world/robot/friction_cone"
        self.forceRadius = 0.015
        self.forceLength = 0.5
        self.forceColor = [1.0, 0.0, 1.0, 1.0]
        self.frictionConeScale = 0.2
        self.frictionConeColor = [0.0, 0.4, 0.79, 0.5]
        self.activeContacts = {}
        self.activeThrust = {}
        self.frictionMu = {}
        self.frameTrajColor = {}
        self.frameTrajLineWidth = 10
        self.x_axis = np.array([1.0, 0.0, 0.0])
        self.y_axis = np.array([0.0, 1.0, 0.0])
        self.z_axis = np.array([0.0, 0.0, 1.0])
        self.totalWeight = sum(
            m.mass for m in self.robot.model.inertias
        ) * np.linalg.norm(self.robot.model.gravity.linear)
        self._init = False

    def displayFromSolver(self, solver, factor=1.0):
        if not self._init:
            self.init(solver)
        dts = self.getDurationSequenceFromSolver(solver)
        fs = self.getForceTrajectoryFromSolver(solver)
        ps = self.getFrameTrajectoryFromSolver(solver)
        rs = self.getThrustTrajectoryFromSolver(solver)
        models = [*solver.problem.runningModels.tolist(), solver.problem.terminalModel]
        dts = [m.dt if hasattr(m, "differential") else 0.0 for m in models]
        self.display(solver.xs, dts, rs, ps, [], fs, [], dts, factor)

    def display(
        self, xs, us=[], rs=[], ps=[], pds=[], fs=[], ss=[], dts=[], factor=1.0
    ):
        if ps:
            self.displayFramePoses(ps)
        if not dts:
            dts = [0.0] * len(xs)
        S = 1 if self.rate <= 0 else max(len(xs) / self.rate, 1)
        for i, x in enumerate(xs):
            if not i % S:
                if rs:
                    if i != len(xs) - 1:
                        for r in rs[i]:
                            # Display the thrust forces
                            self.displayThrustForce(r)
                    else:
                        for key, _ in self.activeThrust.items():
                            thrustName = self.thrustGroup + "/" + key
                            self.setVisibility(thrustName, False)
                if fs:
                    self.activeContacts = {
                        k: False for k, c in self.activeContacts.items()
                    }
                    for f in fs[i]:
                        key = f["key"]
                        # Display the contact forces
                        self.displayContactForce(f)
                        # Display the friction cones
                        self.displayFrictionCone(f)
                        self.activeContacts[key] = True
                for key, c in self.activeContacts.items():
                    if c is False:
                        forceName = self.forceGroup + "/" + key
                        coneName = self.frictionGroup + "/" + key
                        self.setVisibility(forceName, False)
                        self.setVisibility(coneName, False)
                self.robot.display(x[: self.robot.nq])
                time.sleep(dts[i] * factor)

    def init(self, solver):
        frameNames, thrusters = [], []
        self.frameTrajNames = []
        models = [*solver.problem.runningModels.tolist(), solver.problem.terminalModel]
        datas = [*solver.problem.runningDatas.tolist(), solver.problem.terminalData]
        for i, data in enumerate(datas):
            model = models[i]
            if self._hasContacts(data):
                contact_model, _ = self._getContactModelAndData(model, data)
                for c in contact_model.todict().values():
                    if hasattr(c, "contact"):
                        name = self.robot.model.frames[c.contact.id].name
                    elif hasattr(c, "impact"):
                        name = self.robot.model.frames[c.impact.id].name
                    if name not in frameNames:
                        frameNames.append(name)
            if hasattr(model, "differential"):
                if isinstance(
                    model.differential.actuation, ActuationModelFloatingBaseThrusters
                ):
                    thrusters.append(model.differential.actuation.thrusters)
        for n in frameNames:
            frameId = self.robot.model.getFrameId(n)
            parentId = self.robot.model.frames[frameId].parentJoint
            self.activeContacts[str(parentId)] = True
            self.frictionMu[str(parentId)] = 0.7
            self.frameTrajNames.append(str(frameId))
        rng = np.random.default_rng()
        for fr in self.frameTrajNames:
            self.frameTrajColor[fr] = list(
                np.hstack([rng.choice(range(256), size=3) / 256.0, 1.0])
            )
        for thrust in thrusters:
            for i, _ in enumerate(thrust):
                frameName = self.robot.model.frames[2].name
                frameId = self.robot.model.getFrameId(frameName)
                self.activeThrust[str(i)] = True
        self._addForceArrows()
        self._addFrameCurves()
        self._addThrustArrows()
        self._addFrictionCones()
        self._init = True

    @abstractmethod
    def setVisibility(self, name, status):
        """Display the frame pose."""
        raise NotImplementedError("Not implemented yet.")

    @abstractmethod
    def displayFramePoses(self, ps):
        """Display the frame pose"""
        raise NotImplementedError("Not implemented yet.")

    @abstractmethod
    def displayContactForce(self, f):
        """Display the contact force"""
        raise NotImplementedError("Not implemented yet.")

    @abstractmethod
    def displayThrustForce(self, r):
        """Display the thrust force"""
        raise NotImplementedError("Not implemented yet.")

    def getTimeSequenceFromSolver(self, solver):
        dts = self.getDurationSequenceFromSolver(solver)
        return self._getTimeSequence(dts)

    def getDurationSequenceFromSolver(self, solver):
        return [
            m.dt if hasattr(m, "differential") else 0.0
            for m in solver.problem.runningModels
        ]

    def getJointTorquesTrajectoryFromSolver(self, solver):
        us = []
        for i in range(solver.problem.T):
            data = solver.problem.runningDatas[i]
            if hasattr(data, "differential"):
                us.append(data.differential.multibody.actuation.tau[self._nv_root :])
            elif isinstance(data, ActionDataImpulseFwdDynamics):
                us.append(
                    np.zeros(solver.problem.runningModels[i].state.nv - self._nv_root)
                )
            else:
                us.append(solver.us[i][self._nv_root :])
        nu = solver.problem.runningModels[i].state.nv - self._nv_root
        us.append(np.zeros(nu + 1)[1:])  # TODO: temporal patch to fix bug in pybind11
        return us

    def getForceTrajectoryFromSolver(self, solver):
        if len(self.frameTrajNames) == 0:
            return None
        fs = []
        models = [*solver.problem.runningModels.tolist(), solver.problem.terminalModel]
        datas = [*solver.problem.runningDatas.tolist(), solver.problem.terminalData]
        for i, data in enumerate(datas):
            model = models[i]
            if self._hasContacts(data):
                contact_model, contact_data = self._getContactModelAndData(model, data)
                cost_model = self._getCostModel(model)
                fs.append(
                    self._getForceInformation(
                        model.state, contact_model, contact_data, cost_model
                    )
                )
        return fs

    def getThrustTrajectoryFromSolver(self, solver):
        fs = []
        for i, model in enumerate(solver.problem.runningModels.tolist()):
            data = solver.problem.runningDatas[i]
            if hasattr(model, "differential"):
                if isinstance(
                    model.differential.actuation, ActuationModelFloatingBaseThrusters
                ):
                    fc = []
                    ui = solver.us[i]
                    for t, thrust in enumerate(model.differential.actuation.thrusters):
                        frameName = self.robot.model.frames[2].name
                        frameId = self.robot.model.getFrameId(frameName)
                        pinocchio.updateFramePlacement(
                            model.differential.state.pinocchio,
                            data.differential.pinocchio,
                            frameId,
                        )
                        oMb = data.differential.pinocchio.oMf[frameId]
                        oMf = oMb.act(thrust.pose)
                        force = ui[t]
                        fc.append(
                            {
                                "key": str(t),
                                "oMf": oMf,
                                "f": force,
                                "type": thrust.type,
                            }
                        )
                    fs.append(fc)
        return fs

    def getFrameTrajectoryFromSolver(self, solver):
        if len(self.frameTrajNames) == 0:
            return None
        ps = {fr: [] for fr in self.frameTrajNames}
        models = [*solver.problem.runningModels.tolist(), solver.problem.terminalModel]
        datas = [*solver.problem.runningDatas.tolist(), solver.problem.terminalData]
        for key, p in ps.items():
            frameId = int(key)
            for i, data in enumerate(datas):
                model = models[i]
                if hasattr(data, "differential"):
                    # Update the frame placement if there is not contact.
                    # Note that, in non-contact cases, the action model does not compute
                    # it for efficiency reason.
                    if isinstance(data.differential, StdVec_DiffActionData):
                        differential = data.differential[0]
                    else:
                        differential = data.differential
                    if (
                        len(differential.multibody.contacts.contacts.todict().items())
                        == 0
                    ):
                        pinocchio.updateFramePlacement(
                            model.differential.state.pinocchio,
                            differential.multibody.pinocchio,
                            frameId,
                        )
                    pose = differential.multibody.pinocchio.oMf[frameId]
                    p.append(np.asarray(pose.translation.T).reshape(-1).tolist())
                elif isinstance(data, ActionDataImpulseFwdDynamics):
                    pose = data.multibody.pinocchio.oMf[frameId]
                    p.append(np.asarray(pose.translation.T).reshape(-1).tolist())
        return ps

    def _getTimeSequence(self, dts):
        t, ts = 0.0, [0.0]
        for dt in dts:
            t += dt
            ts.append(t)
        return ts

    def _hasContacts(self, data):
        if hasattr(data, "differential"):
            if isinstance(
                data.differential,
                DifferentialActionDataContactFwdDynamics,
            ) or isinstance(
                data.differential,
                DifferentialActionModelContactInvDynamics.DifferentialActionDataContactInvDynamics,
            ):
                return True
        elif isinstance(data, ActionDataImpulseFwdDynamics):
            return True

    def _getContactModelAndData(self, model, data):
        if hasattr(data, "differential"):
            if isinstance(
                data.differential,
                DifferentialActionDataContactFwdDynamics,
            ) or isinstance(
                data.differential,
                DifferentialActionModelContactInvDynamics.DifferentialActionDataContactInvDynamics,
            ):
                return (
                    model.differential.contacts.contacts,
                    data.differential.multibody.contacts.contacts,
                )
            elif isinstance(data.differential, StdVec_DiffActionData) and (
                isinstance(
                    data.differential,
                    DifferentialActionDataContactFwdDynamics,
                )
                or isinstance(
                    data.differential,
                    DifferentialActionModelContactInvDynamics.DifferentialActionDataContactInvDynamics,
                )
            ):
                return (
                    model.differential[0].contacts.contacts,
                    data.differential[0].multibody.contacts.contacts,
                )
        elif isinstance(data, ActionDataImpulseFwdDynamics):
            return model.impulses.impulses, data.multibody.impulses.impulses

    def _getCostModel(self, model):
        if hasattr(model, "differential"):
            return model.differential.costs.costs
        elif isinstance(model, ActionModelImpulseFwdDynamics):
            return model.costs.costs

    def _getForceInformation(self, state, contact_model, contact_data, cost_model):
        fc = []
        for key, contact in contact_data.todict().items():
            if contact_model[key].active:
                joint = state.pinocchio.frames[contact.frame].parentJoint
                oMf = contact.pinocchio.oMi[joint] * contact.jMf
                fiMo = pinocchio.SE3(
                    contact.pinocchio.oMi[joint].rotation.T,
                    contact.jMf.translation,
                )
                force = fiMo.actInv(contact.fext)
                R = np.eye(3)
                mu = 0.7
                for c in cost_model.todict().values():
                    if isinstance(
                        c.cost.residual,
                        ResidualModelContactFrictionCone,
                    ):
                        if contact.frame == c.cost.residual.id:
                            R = c.cost.residual.reference.R
                            mu = c.cost.residual.reference.mu
                            continue
                fc.append(
                    {
                        "key": str(joint),
                        "oMf": oMf,
                        "f": force,
                        "R": R,
                        "mu": mu,
                    }
                )
        return fc


class GepettoDisplay(DisplayAbstract):
    def __init__(
        self,
        robot,
        rate=-1,
        freq=1,
        cameraTF=None,
        floor=True,
        frameNames=None,
        visibility=False,
    ):
        DisplayAbstract.__init__(self, robot, rate, freq)
        if frameNames is not None:
            print("Deprecated. Do not pass frameNames")

        # Visuals properties
        self.fullVisibility = visibility
        self.floorGroup = "world/floor"
        self.backgroundColor = [1.0, 1.0, 1.0, 1.0]
        self.floorScale = [0.5, 0.5, 0.5]
        self.floorColor = [0.7, 0.7, 0.7, 1.0]
        self.frictionConeRays = True
        self._addRobot()
        self._setBackground()
        if cameraTF is not None:
            self.robot.viewer.gui.setCameraTransform(self.robot.viz.windowID, cameraTF)
        if floor:
            self._addFloor()
        self.robot.viewer.gui.createGroup(self.forceGroup)
        self.robot.viewer.gui.createGroup(self.frictionGroup)
        self.robot.viewer.gui.createGroup(self.frameTrajGroup)
        self._addForceArrows()
        self._addThrustArrows()
        self._addFrameCurves()
        self._addFrictionCones()

    def setVisibility(self, name, status):
        self.robot.viewer[name].set_property("visible", status)
        if status:
            self.robot.viewer.gui.setVisibility(name, "ON")
        else:
            self.robot.viewer.gui.setVisibility(name, "OFF")

    def displayFramePoses(self, ps):
        for key, p in ps.items():
            self.robot.viewer.gui.setCurvePoints(self.frameTrajGroup + "/" + key, p)

    def displayContactForce(self, f):
        key, pose, wrench = f["key"], f["oMf"], f["f"]
        forceMagnitud = np.linalg.norm(wrench.linear) / self.totalWeight
        R = rotationMatrixFromTwoVectors(self.x_axis, wrench.linear)
        forcePose = pinocchio.SE3ToXYZQUATtuple(pinocchio.SE3(R, pose.translation))
        forceName = self.forceGroup + "/" + key
        self.robot.viewer.gui.applyConfiguration(forceName, forcePose)
        self.robot.viewer.gui.setVector3Property(
            forceName, "Scale", [1.0 * forceMagnitud, 1.0, 1.0]
        )
        self.robot.viewer.gui.setVisibility(forceName, "ON")

    def displayThrustForce(self, f):
        key, pose, thrust = f["key"], f["oMf"], f["f"]
        wrench = pose.act(pinocchio.Force(np.array([0, 0, thrust]), np.zeros(3)))
        forceMagnitud = np.linalg.norm(wrench.linear) / self.totalWeight
        forcePose = pinocchio.SE3ToXYZQUATtuple(pinocchio.SE3(R, pose.translation))
        thrustName = self.thrustGroup + "/" + key
        self.robot.viewer.gui.applyConfiguration(thrustName, forcePose)
        self.robot.viewer.gui.setVector3Property(
            thrustName, "Scale", [1.0 * forceMagnitud, 1.0, 1.0]
        )

    def displayFrictionCone(self, f):
        key, pose, mu = f["key"], f["oMf"], f["mu"]
        position = pose
        position.rotation = f["R"]
        frictionName = self.frictionGroup + "/" + key
        self._setConeMu(key, mu)
        self.robot.viewer.gui.applyConfiguration(
            frictionName,
            list(np.array(pinocchio.SE3ToXYZQUAT(position)).squeeze()),
        )
        self.robot.viewer.gui.setVisibility(frictionName, "ON")

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
            self.robot.viewer.gui.addArrow(
                forceName, self.forceRadius, self.forceLength, self.forceColor
            )
            self.robot.viewer.gui.setFloatProperty(forceName, "Alpha", 1.0)
        if self.fullVisibility:
            self.robot.viewer.gui.setVisibility(self.forceGroup, "ALWAYS_ON_TOP")

    def _addFrictionCones(self):
        for key in self.activeContacts:
            self._createCone(key, self.frictionConeScale, mu=0.7)

    def _addFrameCurves(self):
        for key in self.frameTrajNames:
            frameName = self.frameTrajGroup + "/" + key
            self.robot.viewer.gui.addCurve(
                frameName,
                [np.array([0.0, 0.0, 0.0]).tolist()] * 2,
                self.frameTrajColor[key],
            )
            self.robot.viewer.gui.setCurveLineWidth(frameName, self.frameTrajLineWidth)
            if self.fullVisibility:
                self.robot.viewer.gui.setVisibility(frameName, "ALWAYS_ON_TOP")

    def _createCone(self, coneName, scale=1.0, mu=0.7):
        m_generatrices = np.matrix(np.empty([3, 4]))
        m_generatrices[:, 0] = np.matrix([mu, mu, 1.0]).T
        m_generatrices[:, 0] = m_generatrices[:, 0] / np.linalg.norm(
            m_generatrices[:, 0]
        )
        m_generatrices[:, 1] = m_generatrices[:, 0]
        m_generatrices[0, 1] *= -1.0
        m_generatrices[:, 2] = m_generatrices[:, 0]
        m_generatrices[:2, 2] *= -1.0
        m_generatrices[:, 3] = m_generatrices[:, 0]
        m_generatrices[1, 3] *= -1.0

        v = [[0.0, 0.0, 0.0]]
        for k in range(m_generatrices.shape[1]):
            v.append(m_generatrices[:3, k].T.tolist()[0])
        v.append(m_generatrices[:3, 0].T.tolist()[0])
        coneGroup = self.frictionGroup + "/" + coneName
        self.robot.viewer.gui.createGroup(coneGroup)

        meshGroup = coneGroup + "/cone"
        self.robot.viewer.gui.addCurve(meshGroup, v, self.frictionConeColor)
        self.robot.viewer.gui.setCurveMode(meshGroup, "TRIANGLE_FAN")
        if self.frictionConeRays:
            lineGroup = coneGroup + "/lines"
            self.robot.viewer.gui.createGroup(lineGroup)
            for k in range(m_generatrices.shape[1]):
                self.robot.viewer.gui.addLine(
                    lineGroup + "/" + str(k),
                    [0.0, 0.0, 0.0],
                    m_generatrices[:3, k].T.tolist()[0],
                    self.frictionConeColor,
                )
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
    def __init__(
        self,
        robot,
        rate=-1,
        freq=1,
        cameraTF=None,
        floor=True,
        frameNames=None,
        visibility=True,
    ):
        DisplayAbstract.__init__(self, robot, rate, freq)
        if frameNames is not None:
            print("Deprecated. Do not pass frameNames")
        robot.setVisualizer(
            MeshcatVisualizer(
                model=self.robot.model,
                collision_model=self.robot.collision_model,
                visual_model=self.robot.visual_model,
            )
        )
        if cameraTF is not None and hasattr(self.robot.viz, "viewer"):
            self.robot.viewer["/Cameras/default"].set_transform(
                pinocchio.XYZQUATToSE3(cameraTF).homogeneous
            )
        self._addRobot(visibility)
        self._addForceArrows()
        self._addThrustArrows()
        self._addFrictionCones()

    def setVisibility(self, name, status):
        self.robot.viewer[name].set_property("visible", status)

    def displayFramePoses(self, ps):
        for key in ps.keys():
            vertices = np.array(ps[key]).T
            self._addFrameCurves(key, vertices)

    def displayContactForce(self, f):
        key, pose, wrench = f["key"], f["oMf"], f["f"]
        forceMagnitud = np.linalg.norm(wrench.linear) / self.totalWeight
        R = rotationMatrixFromTwoVectors(self.y_axis, wrench.linear)
        forcePose = pinocchio.SE3(
            R,
            pose.translation
            + np.dot(
                R,
                np.array([0.0, forceMagnitud * self.forceLength / 2, 0.0]),
            ),
        )
        forceName = self.forceGroup + "/" + key
        self.robot.viewer[forceName].set_property("visible", False)
        self.robot.viewer[forceName].set_transform(forcePose.homogeneous)
        self.robot.viewer[forceName].set_property("scale", [1.0, forceMagnitud, 1.0])
        self.robot.viewer[forceName].set_property("visible", True)

    def displayThrustForce(self, f):
        key, pose, thrust = f["key"], f["oMf"], f["f"]
        wrench = pose.act(pinocchio.Force(np.array([0, 0, thrust]), np.zeros(3)))
        forceMagnitud = np.linalg.norm(wrench.linear) / self.totalWeight
        R = rotationMatrixFromTwoVectors(self.y_axis, wrench.linear)
        forcePose = pinocchio.SE3(
            R,
            pose.translation
            + np.dot(
                R,
                np.array([0.0, forceMagnitud * self.forceLength / 2, 0.0]),
            ),
        )
        thrustName = self.thrustGroup + "/" + key
        self.robot.viewer[thrustName].set_property("visible", False)
        self.robot.viewer[thrustName].set_transform(forcePose.homogeneous)
        self.robot.viewer[thrustName].set_property("scale", [1.0, forceMagnitud, 1.0])
        self.robot.viewer[thrustName].set_property("visible", True)

    def displayFrictionCone(self, f):
        key, pose, mu = f["key"], f["oMf"], f["mu"]
        R = pinocchio.utils.rpyToMatrix(-np.pi / 2, 0.0, 0.0)
        conePose = pinocchio.SE3(
            R,
            pose.translation
            + np.dot(R, np.array([0.0, -self.frictionConeScale / 2, 0.0])),
        ).homogeneous
        coneName = self.frictionGroup + "/" + key
        self.robot.viewer[coneName].set_property("radiusBottom", mu)
        self.robot.viewer[coneName].set_transform(conePose)
        self.robot.viewer[coneName].set_property("visible", True)

    def _addRobot(self, openWindow):
        self.robot.initViewer(open=openWindow)
        self.robot.loadViewerModel(rootNodeName="robot")

    def _addForceArrows(self):
        import meshcat.geometry as g

        meshColor = g.MeshLambertMaterial(
            color=self._rgbToHexColor(self.forceColor[:3]), reflectivity=0.8
        )
        for key in self.activeContacts:
            forceName = self.forceGroup + "/" + key
            self.robot.viewer[forceName].set_object(
                g.Cylinder(self.forceLength, self.forceRadius), meshColor
            )

    def _addThrustArrows(self):
        import meshcat.geometry as g

        meshColor = g.MeshLambertMaterial(
            color=self._rgbToHexColor(self.forceColor[:3]), reflectivity=0.8
        )
        for key in self.activeThrust:
            thrustName = self.thrustGroup + "/" + key
            self.robot.viewer[thrustName].set_object(
                g.Cylinder(self.forceLength, self.forceRadius), meshColor
            )

    def _addFrictionCones(self):
        import meshcat.geometry as g

        meshColor = g.MeshLambertMaterial(
            color=self._rgbToHexColor(self.frictionConeColor[:3]),
            reflectivity=0.8,
            opacity=0.2,
            transparent=True,
        )
        for key in self.activeContacts:
            coneName = self.frictionGroup + "/" + key
            mu = self.frictionMu[key]
            self.robot.viewer[coneName].set_object(
                g.Cylinder(
                    self.frictionConeScale, None, 0.0, mu * self.frictionConeScale
                ),
                meshColor,
            )

    def _addFrameCurves(self, key=None, vertices=None):
        if key is None and vertices is None:
            return
        import meshcat.geometry as g

        frameName = self.frameTrajGroup + "/" + key
        meshColor = g.LineBasicMaterial(
            color=self._rgbToHexColor(self.frameTrajColor[key][:3]), linewidth=3.0
        )
        self.robot.viewer[frameName].set_object(
            g.Line(g.PointsGeometry(vertices), meshColor)
        )

    def _rgbToHexColor(self, rgbColor):
        return "0x{:02x}{:02x}{:02x}".format(
            *tuple(np.rint(255 * np.array(rgbColor)).astype(int))
        )


class RvizDisplay(DisplayAbstract):
    def __init__(
        self,
        robot,
        rate=-1,
        freq=1,
    ):
        DisplayAbstract.__init__(self, robot, rate, freq)
        # Disable garbage collection for better timings.
        import gc

        gc.disable()
        # Import ROS modules
        self.ROS_VERSION = int(os.environ["ROS_VERSION"])
        if self.ROS_VERSION == 2:
            import rclpy
        else:
            import roslaunch
            import rospy
        import crocoddyl_ros
        from urdf_parser_py.urdf import URDF

        # Init the ROS node and publishers
        if self.ROS_VERSION == 2:
            if not rclpy.ok():
                rclpy.init()
        else:
            self.roscore = roslaunch.parent.ROSLaunchParent(
                "crocoddyl_display", [], is_core=True
            )
            self.roscore.start()
            rospy.init_node("crocoddyl_display", anonymous=True)
            rospy.set_param("use_sim_time", True)
            rospy.set_param(
                "robot_description", URDF.from_xml_file(robot.urdf).to_xml_string()
            )  # TODO: hard code robot.urdf because we cannot convert a Pinocchio model into an URDF
            filename = os.path.join(
                os.path.dirname(
                    os.path.abspath(inspect.getfile(inspect.currentframe()))
                ),
                "crocoddyl.rviz",
            )
            rviz_args = [
                os.path.join(
                    os.path.dirname(
                        os.path.abspath(inspect.getfile(inspect.currentframe()))
                    ),
                    "crocoddyl.launch",
                ),
                "filename:='{filename}'".format_map(locals()),
            ]
            roslaunch_args = rviz_args[1:]
            roslaunch_file = [
                (
                    roslaunch.rlutil.resolve_launch_arguments(rviz_args)[0],
                    roslaunch_args,
                )
            ]
            self.rviz = roslaunch.parent.ROSLaunchParent(
                "crocoddyl_display", roslaunch_file, is_core=False
            )
            self.rviz.start()
        self._wsPublisher = crocoddyl_ros.WholeBodyStateRosPublisher(
            robot.model, "whole_body_state", "map"
        )
        self._wtPublisher = crocoddyl_ros.WholeBodyTrajectoryRosPublisher(
            robot.model, "whole_body_trajectory", "map"
        )
        root_id = crocoddyl_ros.getRootJointId(self.robot.model)
        self._nv_root = (
            self.robot.model.joints[root_id].nv
            if self.robot.model.frames[root_id].name != "universe"
            else 0
        )

    def __del__(self):
        if self.ROS_VERSION == 1:
            self.roscore.shutdown()
            self.rviz.shutdown()

    def displayFromSolver(self, solver, factor=1.0):
        if not self._init:
            self.init(solver)
        xs = solver.xs
        us = self.getJointTorquesTrajectoryFromSolver(solver)
        dts = self.getDurationSequenceFromSolver(solver)
        models = [*solver.problem.runningModels.tolist(), solver.problem.terminalModel]
        dts = [m.dt if hasattr(m, "differential") else 0.0 for m in models]
        ps, pds = self.getFrameTrajectoryFromSolver(solver)
        fs, ss = self.getForceTrajectoryFromSolver(solver)
        self.display(xs, us, dts, ps, pds, fs, ss, dts, factor)

    def display(
        self, xs, us=[], rs=[], ps=[], pds=[], fs=[], ss=[], dts=[], factor=1.0
    ):
        nq = self.robot.model.nq
        if not dts:
            dts = [0.01] * len(xs)
        ts = self._getTimeSequence(dts)[:-1]
        if len(ps) != 0:
            self._wtPublisher.publish(ts, xs, us, ps, pds, fs, ss)
        else:
            self._wtPublisher.publish(ts, xs, us)
        for i in range(len(xs)):
            t, x, u = ts[i], xs[i], us[i]
            if len(ps) != 0:
                p, pd = ps[i], pds[i]
                f, s = fs[i], ss[i]
                self._wsPublisher.publish(t, x[:nq], x[nq:], u, p, pd, f, s)
            else:
                self._wsPublisher.publish(t, x[:nq], x[nq:], u)
            time.sleep(dts[i] * factor)

    def getFrameTrajectoryFromSolver(self, solver):
        ps, pds = [], []
        datas = [*solver.problem.runningDatas.tolist(), solver.problem.runningDatas[-1]]
        for i, data in enumerate(datas):
            if self._hasContacts(data):
                pinocchio_data = self._getPinocchioData(data)
                ps.append(self._get_pc(pinocchio_data))
                pds.append(self._get_pdc(pinocchio_data))
        return ps, pds

    def getForceTrajectoryFromSolver(self, solver):
        fs, ss = [], []
        models = [
            *solver.problem.runningModels.tolist(),
            solver.problem.runningModels[-1],
        ]
        datas = [*solver.problem.runningDatas.tolist(), solver.problem.runningDatas[-1]]
        for i, data in enumerate(datas):
            model = models[i]
            if self._hasContacts(data):
                _, contact_data = self._getContactModelAndData(model, data)
                fs.append(self._get_fc(contact_data))
                ss.append(self._get_sc(contact_data))
        return fs, ss

    def setVisibility(self, name, status):
        pass

    def displayFramePoses(self, ps):
        pass

    def displayContactForce(self, f):
        pass

    def displayThrustForce(self, r):
        pass

    def _addForceArrows(self):
        pass

    def _addThrustArrows(self):
        pass

    def _addFrameCurves(self):
        pass

    def _addFrictionCones(self):
        pass

    def _getPinocchioData(self, data):
        if hasattr(data, "differential"):
            if hasattr(data.differential, "multibody"):
                return data.differential.multibody.pinocchio
            elif isinstance(data.differential, StdVec_DiffActionData):
                if hasattr(data.differential[0], "multibody"):
                    return data.differential[0].multibody.pinocchio
        elif isinstance(data, ActionDataImpulseFwdDynamics):
            return data.multibody.pinocchio

    def _get_pc(self, pinocchio_data):
        if len(self.frameTrajNames) == 0:
            return None
        popt = dict()
        for frame in self.frameTrajNames:
            frame_id = int(frame)
            name = self.robot.model.frames[frame_id].name
            pinocchio.updateFramePlacement(self.robot.model, pinocchio_data, frame_id)
            popt[name] = pinocchio_data.oMf[frame_id]
        return popt

    def _get_pdc(self, pinocchio_data):
        if len(self.frameTrajNames) == 0:
            return None
        pdopt = dict()
        for frame in self.frameTrajNames:
            frame_id = int(frame)
            name = self.robot.model.frames[frame_id].name
            v = pinocchio.getFrameVelocity(
                self.robot.model,
                pinocchio_data,
                frame_id,
                pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            pdopt[name] = v
        return pdopt

    def _get_fc(self, contact_data):
        import crocoddyl_ros

        if len(self.frameTrajNames) == 0:
            return None
        fc = dict()
        for frame in self.frameTrajNames:
            frame_id = int(frame)
            name = self.robot.model.frames[frame_id].name
            fc[name] = [
                pinocchio.Force.Zero(),
                crocoddyl_ros.ContactType.LOCOMOTION,
                crocoddyl_ros.ContactStatus.SEPARATION,
            ]
        for contact in contact_data:
            force = contact.data().f
            frame_id = contact.data().frame
            name = self.robot.model.frames[frame_id].name
            fc[name] = [
                force,
                crocoddyl_ros.ContactType.LOCOMOTION,
                crocoddyl_ros.ContactStatus.STICKING,
            ]
        return fc

    def _get_sc(self, contact_data):
        sc = dict()
        self.mu = 0.7
        for frame in self.frameTrajNames:
            frame_id = int(frame)
            name = self.robot.model.frames[frame_id].name
            sc[name] = [np.array([0.0, 0.0, 1.0]), self.mu]
        return sc


class CallbackDisplay(CallbackAbstract):
    def __init__(self, display):
        CallbackAbstract.__init__(self)
        self.visualization = display

    def __call__(self, solver):
        if (solver.iter + 1) % self.visualization.freq:
            return
        self.visualization.displayFromSolver(solver)


class CallbackLogger(CallbackAbstract):
    def __init__(self):
        CallbackAbstract.__init__(self)
        self.xs = []
        self.us = []
        self.fs = []
        self.iters = []
        self.costs = []
        self.stops = []
        self.grads = []
        self.pregs = []
        self.dregs = []
        self.steps = []
        self.ffeass = []
        self.hfeass = []

    def __call__(self, solver):
        import copy

        self.xs = copy.copy(solver.xs)
        self.us = copy.copy(solver.us)
        self.fs.append(copy.copy(solver.fs))
        self.iters.append(solver.iter)
        self.costs.append(solver.cost)
        self.stops.append(solver.stoppingCriteria())
        self.grads.append(-solver.expectedImprovement()[1].item())
        self.pregs.append(solver.preg)
        self.dregs.append(solver.dreg)
        self.steps.append(solver.stepLength)
        self.ffeass.append(solver.ffeas)
        self.hfeass.append(solver.hfeas)


def plotOCSolution(xs=None, us=None, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Getting the state and control trajectories
    if xs is not None:
        xsPlotIdx = 111
        nx = xs[0].shape[0]
        X = [0.0] * nx
        for i in range(nx):
            X[i] = [x[i] for x in xs]
    if us is not None:
        usPlotIdx = 111
        nu = us[0].shape[0]
        U = [0.0] * nu
        for i in range(nu):
            U[i] = [u[i] if u.shape[0] != 0 else 0 for u in us]
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


def plotFeasibility(ffeass, hfeass, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.figure(figIndex, figsize=(6.4, 8))

    # Plotting the feasibility
    plt.ylabel("feasibiltiy")
    plt.plot(ffeass)
    plt.plot(hfeass)
    plt.plot([max(ffeas, hfeas) for ffeas, hfeas in zip(ffeass, hfeass)])
    plt.title(figTitle, fontsize=14)
    plt.xlabel("iteration")
    plt.yscale("log")
    plt.legend(["dynamic", "equality", "total"])
    if show:
        plt.show()


def plotConvergence(
    costs, muLM, muV, gamma, theta, alpha, figIndex=1, show=True, figTitle=""
):
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

    data = {
        "costs": costs,
        "muLM": muLM,
        "muV": muV,
        "gamma": gamma,
        "theta": theta,
        "alpha": alpha,
    }
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
        "primal-reg": log.pregs,
        "dual-reg": log.dregs,
        "stops": log.stops,
        "grads": log.grads,
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
