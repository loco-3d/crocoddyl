import copy
import sys

import numpy as np
import pinocchio

import crocoddyl


class HumanoidLocoManipulation:
    def __init__(
        self, q0, robot_model, right_foot, left_foot, right_hand, left_hand, fwddyn=True
    ):
        self.robot_model = robot_model
        self.robot_data = robot_model.createData()
        self.RF_name = right_foot
        self.LF_name = left_foot
        self.RH_name = right_hand
        self.LH_name = left_hand
        self._fwddyn = fwddyn
        self.state = crocoddyl.StateMultibody(self.robot_model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.q0 = q0
        self.x0 = np.concatenate([self.q0, np.zeros(self.state.nv)])
        self.qref = copy.deepcopy(
            self.robot_model.referenceConfigurations["half_sitting"]
        )
        self.xref = np.concatenate([self.qref, np.zeros(self.state.nv)])
        self.mu = 0.7
        self.comArea = np.array([0.1, 0.05])
        self.dt = 3e-2

        maxfloat = sys.float_info.max
        self.xlb = np.hstack(
            [
                -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
                robot_model.lowerPositionLimit[7:],
                -maxfloat * np.ones(self.state.nv),
            ]
        )
        self.xub = np.hstack(
            [
                maxfloat * np.ones(6),  # dimension of the SE(3) manifold
                robot_model.upperPositionLimit[7:],
                maxfloat * np.ones(self.state.nv),
            ]
        )
        nv = self.state.nv
        self.stateWeights = np.array(
            [0] * 3 + [10.0] * 3 + [0.01] * (nv - 6) + [20.0] * nv
        )

    def createModel(
        self,
        qref=None,
        footContacts=list(),
        handContacts=list(),
        bodiesTarget=None,
        handsTarget=dict(),
        feetTarget=dict(),
        switch=False,
        constraint=False,
    ):
        if self._fwddyn:
            nu = self.actuation.nu if switch is False else 0
        else:
            nu = (
                self.state.nv + 6 * len(footContacts + handContacts)
                if switch is False
                else 0
            )
        costs = crocoddyl.CostModelSum(self.state, nu)
        constraints = crocoddyl.ConstraintModelManager(self.state, nu)
        if not switch:
            contacts = crocoddyl.ContactModelMultiple(self.state, nu)
        else:
            impulses = crocoddyl.ImpulseModelMultiple(self.state)
        # Cost for body target
        if bodiesTarget is not None:
            for name, Mref in bodiesTarget.items():
                frame_id = self.robot_model.getFrameId(name)
                # Cost for target reaching: bodies
                if isinstance(Mref, (np.ndarray, np.generic)):
                    costs.addCost(
                        name + "_pose",
                        self._createFrameRotationCost(name, Mref, np.ones(3), nu),
                        1e3,
                    )
                elif isinstance(Mref, pinocchio.SE3):
                    costs.addCost(
                        name + "_pose",
                        self._createFramePlacementCost(name, Mref, np.ones(6), nu),
                        1e3,
                    )
        # Cost for self-collision, and for state and control regularization
        costs.addCost("stateLimitsCost", self._createStateLimsCost(nu), 1e3)
        costs.addCost(
            "stateReg", self._createStateRegCost(qref, self.stateWeights, nu), 1e-3
        )
        if not switch:
            costs.addCost("effortReg", self._createEffortRegCost(nu), 1e-4)
            costs.addCost("accelerationReg", self._createAccelerationRegCost(nu), 1e-4)
        for name in footContacts:
            frame_id = self.robot_model.getFrameId(name)
            Mref = pinocchio.SE3.Identity()
            if switch is False:
                contact = crocoddyl.ContactModel6D(
                    self.state,
                    frame_id,
                    Mref,
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                    np.array([0.0, 40.0]),
                )
                contacts.addContact(name + "_contact", contact)
            else:
                impulse = crocoddyl.ImpulseModel6D(
                    self.state, frame_id, pinocchio.LOCAL_WORLD_ALIGNED
                )
                impulses.addImpulse(name + "_contact", impulse)
            # Cost for wrench cone
            costs.addCost(
                name + "_frictionCone",
                self._createWrenchConeCost(
                    name, Mref.rotation, self.mu, self.comArea, nu
                ),
                1e1,
            )
            # Cost for foot-force regularization
            costs.addCost(name + "_forceReg", self._createForceRegCost(name, nu), 1e-5)
        for name in handContacts:
            frame_id = self.robot_model.getFrameId(name)
            Mref = pinocchio.SE3.Identity()
            if not switch:
                contact = crocoddyl.ContactModel6D(
                    self.state,
                    frame_id,
                    Mref,
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                    np.array([0.0, 40.0]),
                )
                contacts.addContact(name + "_contact", contact)
            else:
                impulse = crocoddyl.ImpulseModel6D(
                    self.state, frame_id, pinocchio.LOCAL_WORLD_ALIGNED
                )
                impulses.addImpulse(name + "_contact", impulse)
            # Cost for friction cone
            costs.addCost(
                name + "_frictionCone",
                self._createFrictionConeCost(name, Mref.rotation, self.mu, nu),
                1e1,
            )
            # Cost for hand-force regularization
            costs.addCost(name + "_forceReg", self._createForceRegCost(name, nu), 1e-5)
        for name, Mref in handsTarget.items():
            frame_id = self.robot_model.getFrameId(name)
            # Cost for target reaching: hands
            handTrackingWeights = np.array([1.0] * 3 + [0.1, 0.1, 0.1])
            if constraint:
                hposeResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, frame_id, Mref, nu
                )
                hposeConstraint = crocoddyl.ConstraintModelResidual(
                    self.state, hposeResidual
                )
                constraints.addConstraint(name + "_pose", hposeConstraint)
            else:
                costs.addCost(
                    name + "_pose",
                    self._createFramePlacementCost(name, Mref, handTrackingWeights, nu),
                    1e3,
                )
        for name, Mref in feetTarget.items():
            frame_id = self.robot_model.getFrameId(name)
            # Cost for target reaching: feet
            footTrackingWeights = np.array([1, 1, 1] + [1.0] * 3)
            if constraint:
                fposeResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, frame_id, Mref, nu
                )
                fposeConstraint = crocoddyl.ConstraintModelResidual(
                    self.state, fposeResidual
                )
                constraints.addConstraint(name + "_pose", fposeConstraint)
            else:
                costs.addCost(
                    name + "_pose",
                    self._createFramePlacementCost(name, Mref, footTrackingWeights, nu),
                    1e3,
                )
        # Create the differential action model
        if switch is False:
            if self._fwddyn:
                dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                    self.state, self.actuation, contacts, costs, constraints, 0.0, True
                )
            else:
                dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                    self.state, self.actuation, contacts, costs, constraints
                )
            model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)
        else:
            model = crocoddyl.ActionModelImpulseFwdDynamics(
                self.state, impulses, costs, constraints, 0.0, 0.0, True
            )
        return model

    def createMonkeyBarProblem(self, RH_pose, LH_pose, RF_pose, LF_pose):
        # Problem definition
        Tstand, Treach, Tds, Tss, Tleave = 50, 8, 10, 8, 12
        # Model for reaching the bars
        handsTarget = {self.LH_name: LH_pose[0], self.RH_name: RH_pose[0]}
        standModel = self.createModel(footContacts=[self.LF_name, self.RF_name])
        jumpBarModel = self.createModel()
        graspSwitch = self.createModel(handsTarget=handsTarget, switch=True)
        dsModel = self.createModel(handContacts=[self.LH_name, self.RH_name])
        dsSwitch0 = self.createModel(
            footContacts=[self.LF_name, self.RF_name],
            handsTarget=handsTarget,
            switch=True,
        )
        models = [standModel] * Tstand + [jumpBarModel] * Treach + [graspSwitch]
        models += [dsModel] * Tds + [dsSwitch0]
        for i in range(len(RH_pose) - 1):
            dsModel = self.createModel(handContacts=[self.LH_name, self.RH_name])
            # Model for the right-hand brachiation
            rightHandTarget1 = {self.RH_name: RH_pose[i + 1]}
            rhModel = self.createModel(handContacts=[self.LH_name])
            rhSwitch = self.createModel(
                handContacts=[self.LH_name, self.RH_name],
                handsTarget=rightHandTarget1,
                switch=True,
            )
            models += [rhModel] * Tss + [rhSwitch]
            # Model for the intermediate two-hands brachiation
            handsTarget1 = {self.LH_name: LH_pose[i], self.RH_name: RH_pose[i + 1]}
            dsSwitch1 = self.createModel(handsTarget=handsTarget1, switch=True)
            models += [dsModel] * Tds + [dsSwitch1]
            # Model for the left-hand brachiation
            leftHandTarget2 = {self.LH_name: LH_pose[i + 1]}
            lhModel = self.createModel(handContacts=[self.RH_name])
            lhSwitch = self.createModel(
                handContacts=[self.RH_name], handsTarget=leftHandTarget2, switch=True
            )
            models += [lhModel] * Tss + [lhSwitch]
            # Model for the final two-hands brachiation
            handsTarget3 = {self.LH_name: LH_pose[i + 1], self.RH_name: RH_pose[i + 1]}
            dsSwitch2 = self.createModel(handsTarget=handsTarget3, switch=True)
            models += [dsModel] * Tds + [dsSwitch2]
        # Model for leaving the bars
        handsTermTarget = dict()
        q0 = copy.deepcopy(self.q0)
        q0[:2] = (LF_pose.translation + RF_pose.translation)[:2] / 2
        pinocchio.forwardKinematics(self.robot_model, self.robot_data, q0)
        for name in [self.LH_name, self.RH_name]:
            frame_id = self.robot_model.getFrameId(name)
            pinocchio.updateFramePlacement(self.robot_model, self.robot_data, frame_id)
            handsTermTarget[name] = self.robot_data.oMf[frame_id]
        feetTarget = {self.LF_name: LF_pose, self.RF_name: RF_pose}
        jumpGroundModel = self.createModel()
        landSwitch = self.createModel(feetTarget=feetTarget, switch=True)
        models += [jumpGroundModel] * Tleave + [landSwitch]
        # Model for the resting posture
        restModel = self.createModel(footContacts=[self.LF_name, self.RF_name])
        restSwitch = self.createModel(
            bodiesTarget={"root_joint": pinocchio.SE3(np.eye(3), q0[:3])},
            footContacts=[self.LF_name, self.RF_name],
            handsTarget=handsTermTarget,
            switch=True,
        )
        models += (
            [restModel] * int(Tstand / 2) + [restSwitch] + [restModel] * int(Tstand / 2)
        )
        terminalModel = self.createModel(footContacts=[self.LF_name, self.RF_name])
        return crocoddyl.ShootingProblem(self.x0, models, terminalModel)

    def createMonkeyOneBarProblem(self, RH_pose, LH_pose, RF_pose, LF_pose):
        # Problem definition
        Tstand, Treach = 50, 8
        # Model for reaching the bars
        handsTarget = {self.LH_name: LH_pose[0], self.RH_name: RH_pose[0]}
        standModel = self.createModel(footContacts=[self.LF_name, self.RF_name])
        jumpBarModel = self.createModel()
        models = [standModel] * Tstand + [jumpBarModel] * Treach
        terminalModel = self.createModel(handsTarget=handsTarget, constraint=True)
        return crocoddyl.ShootingProblem(self.x0, models, terminalModel)

    def createHandstandProblem(self, RH_pose, LH_pose, RF_pose, LF_pose):
        # Problem definition
        Tstand, Trotate, Tds, Tss, Tleave = 50, 15, 10, 8, 12
        # Model for reaching the bars
        handsTarget = {self.LH_name: LH_pose[0], self.RH_name: RH_pose[0]}
        standModel = self.createModel(footContacts=[self.LF_name, self.RF_name])
        standSwitch = self.createModel(
            footContacts=[self.LF_name, self.RF_name],
            handsTarget=handsTarget,
            switch=True,
        )
        # Model for handstanding on the first bar
        Rfoot = pinocchio.utils.rpyToMatrix(0, np.pi, 0.0)
        LF_pose0 = pinocchio.SE3(
            Rfoot,
            np.array(
                [
                    LH_pose[0].translation[0],
                    LF_pose.translation[1],
                    LH_pose[0].translation[2] + 2.0,
                ]
            ),
        )
        RF_pose0 = pinocchio.SE3(
            Rfoot,
            np.array(
                [
                    RH_pose[0].translation[0],
                    RF_pose.translation[1],
                    RH_pose[0].translation[2] + 2.0,
                ]
            ),
        )
        feetTarget = {self.LF_name: LF_pose0, self.RF_name: RF_pose0}
        Mref = pinocchio.SE3(pinocchio.utils.rpyToMatrix(0, np.pi, 0.0), np.zeros(3))
        qref = np.concatenate([pinocchio.SE3ToXYZQUAT(Mref), self.q0[7:]])
        rotateBodyModel = self.createModel(
            qref=qref, handContacts=[self.LH_name, self.RH_name]
        )
        graspSwitch = self.createModel(
            qref=qref,
            handContacts=[self.LH_name, self.RH_name],
            feetTarget=feetTarget,
            switch=True,
        )
        dsModel = self.createModel(qref=qref, handContacts=[self.LH_name, self.RH_name])
        models = [standModel] * Tstand + [standSwitch]
        models += [rotateBodyModel] * Trotate + [graspSwitch]
        for i in range(len(RH_pose) - 1):
            dsModel = self.createModel(
                qref=qref, handContacts=[self.LH_name, self.RH_name]
            )
            # Model for the right-hand brachiation
            rightHandTarget1 = {self.RH_name: RH_pose[i + 1]}
            rhModel = self.createModel(qref=qref, handContacts=[self.LH_name])
            rhSwitch = self.createModel(
                qref=qref,
                handContacts=[self.LH_name, self.RH_name],
                handsTarget=rightHandTarget1,
                switch=True,
            )
            models += [rhModel] * Tss + [rhSwitch]
            # Model for the intermediate two-hands brachiation
            handsTarget1 = {self.LH_name: LH_pose[i], self.RH_name: RH_pose[i + 1]}
            dsSwitch1 = self.createModel(
                qref=qref, handsTarget=handsTarget1, switch=True
            )
            models += [dsModel] * Tds + [dsSwitch1]
            # Model for the left-hand brachiation
            leftHandTarget2 = {self.LH_name: LH_pose[i + 1]}
            lhModel = self.createModel(qref=qref, handContacts=[self.RH_name])
            lhSwitch = self.createModel(
                qref=qref,
                handContacts=[self.RH_name],
                handsTarget=leftHandTarget2,
                switch=True,
            )
            models += [lhModel] * Tss + [lhSwitch]
            # Model for the final two-hands brachiation
            handsTarget3 = {self.LH_name: LH_pose[i + 1], self.RH_name: RH_pose[i + 1]}
            if i == len(RH_pose) - 2:
                LF_poseN = pinocchio.SE3(
                    Rfoot,
                    np.array(
                        [
                            LH_pose[i + 1].translation[0],
                            LF_pose.translation[1],
                            LH_pose[i + 1].translation[2] + 2.0,
                        ]
                    ),
                )
                RF_poseN = pinocchio.SE3(
                    Rfoot,
                    np.array(
                        [
                            RH_pose[i + 1].translation[0],
                            RF_pose.translation[1],
                            RH_pose[i + 1].translation[2] + 2.0,
                        ]
                    ),
                )
                feetTarget = {self.LF_name: LF_poseN, self.RF_name: RF_poseN}
                dsSwitch2 = self.createModel(
                    qref=qref,
                    feetTarget=feetTarget,
                    handsTarget=handsTarget3,
                    switch=True,
                )
            else:
                dsSwitch2 = self.createModel(
                    qref=qref, handsTarget=handsTarget3, switch=True
                )
            models += [dsModel] * Tds + [dsSwitch2]
        # Model for leaving the bars
        handsTermTarget = dict()
        q0 = copy.deepcopy(self.q0)
        q0[:2] = (LF_pose.translation + RF_pose.translation)[:2] / 2
        pinocchio.forwardKinematics(self.robot_model, self.robot_data, q0)
        for name in [self.LH_name, self.RH_name]:
            frame_id = self.robot_model.getFrameId(name)
            pinocchio.updateFramePlacement(self.robot_model, self.robot_data, frame_id)
            handsTermTarget[name] = self.robot_data.oMf[frame_id]
        feetTarget = {self.LF_name: LF_pose, self.RF_name: RF_pose}
        jumpGroundModel = self.createModel()
        landSwitch = self.createModel(feetTarget=feetTarget, switch=True)
        models += [jumpGroundModel] * Tleave + [landSwitch]
        # Model for the resting posture
        restModel = self.createModel(footContacts=[self.LF_name, self.RF_name])
        restSwitch = self.createModel(
            bodiesTarget={"root_joint": pinocchio.SE3(np.eye(3), q0[:3])},
            footContacts=[self.LF_name, self.RF_name],
            handsTarget=handsTermTarget,
            switch=True,
        )
        models += (
            [restModel] * int(Tstand / 2) + [restSwitch] + [restModel] * int(Tstand / 2)
        )
        terminalModel = self.createModel(footContacts=[self.LF_name, self.RF_name])
        return crocoddyl.ShootingProblem(self.x0, models, terminalModel)

    def createHandstandEquilibriumProblem(self, RH_pose, LH_pose, RF_pose, LF_pose):
        # Problem definition
        Tstand, Trotate = 50, 15
        # Model for reaching the bars
        handsTarget = {self.LH_name: LH_pose[0], self.RH_name: RH_pose[0]}
        standModel = self.createModel(footContacts=[self.LF_name, self.RF_name])
        standSwitch = self.createModel(
            footContacts=[self.LF_name, self.RF_name],
            handsTarget=handsTarget,
            switch=True,
        )
        # Model for handstanding on the first bar
        Rfoot = pinocchio.utils.rpyToMatrix(0, np.pi, 0.0)
        LF_pose0 = pinocchio.SE3(
            Rfoot,
            np.array(
                [
                    LH_pose[0].translation[0],
                    LF_pose.translation[1],
                    LH_pose[0].translation[2] + 2.0,
                ]
            ),
        )
        RF_pose0 = pinocchio.SE3(
            Rfoot,
            np.array(
                [
                    RH_pose[0].translation[0],
                    RF_pose.translation[1],
                    RH_pose[0].translation[2] + 2.0,
                ]
            ),
        )
        feetTarget = {self.LF_name: LF_pose0, self.RF_name: RF_pose0}
        Mref = pinocchio.SE3(pinocchio.utils.rpyToMatrix(0, np.pi, 0.0), np.zeros(3))
        qref = np.concatenate([pinocchio.SE3ToXYZQUAT(Mref), self.q0[7:]])
        rotateBodyModel = self.createModel(
            qref=qref, handContacts=[self.LH_name, self.RH_name]
        )
        models = [standModel] * Tstand + [standSwitch]
        models += [rotateBodyModel] * Trotate
        terminalModel = self.createModel(
            qref=qref,
            handContacts=[self.LH_name, self.RH_name],
            feetTarget=feetTarget,
            constraint=True,
        )
        return crocoddyl.ShootingProblem(self.x0, models, terminalModel)

    def createFlipProblem(self, distance, front=True):
        # Problem definition
        Tstand, Tstart, Tflipseg, Tend = 40, 2, 3, 20
        # Model for stand and backflip phases
        standModel = self.createModel(
            # bodiesTarget={"root_joint": np.eye(3)},
            footContacts=[self.LF_name, self.RF_name]
        )
        backflipModel = self.createModel()
        # Models for each backflip phase
        models = [standModel] * Tstand
        models += [backflipModel] * Tstart
        Rbody, nphase = np.eye(3), 6
        for i in range(nphase):
            if front:
                Rbody = Rbody @ pinocchio.utils.rpyToMatrix(0, 2 * np.pi / nphase, 0)
            else:
                Rbody = Rbody @ pinocchio.utils.rpyToMatrix(0, -2 * np.pi / nphase, 0)
            Mref = pinocchio.SE3(Rbody, np.zeros(3))
            qref = np.concatenate([pinocchio.SE3ToXYZQUAT(Mref), self.q0[7:]])
            if i == nphase - 1:
                # Compute the feet placement when landing
                if front:
                    jump_distance = np.array([distance, 0.0, 0.0])
                else:
                    jump_distance = np.array([-distance, 0.0, 0.0])
                pinocchio.forwardKinematics(self.robot_model, self.robot_data, self.q0)
                pinocchio.updateFramePlacement(
                    self.robot_model,
                    self.robot_data,
                    self.robot_model.getFrameId(self.LF_name),
                )
                pinocchio.updateFramePlacement(
                    self.robot_model,
                    self.robot_data,
                    self.robot_model.getFrameId(self.RF_name),
                )
                LF_pose = pinocchio.SE3(
                    np.eye(3),
                    self.robot_data.oMf[
                        self.robot_model.getFrameId(self.LF_name)
                    ].translation
                    + jump_distance,
                )
                RF_pose = pinocchio.SE3(
                    np.eye(3),
                    self.robot_data.oMf[
                        self.robot_model.getFrameId(self.RF_name)
                    ].translation
                    + jump_distance,
                )
                feetTarget = {self.LF_name: LF_pose, self.RF_name: RF_pose}
                backflipAngleModel = self.createModel(
                    qref=qref,
                    bodiesTarget={"root_joint": Rbody},
                    feetTarget=feetTarget,
                    switch=True,
                )
            else:
                backflipAngleModel = self.createModel(
                    qref=qref, bodiesTarget={"root_joint": Rbody}
                )
            models += [backflipModel] * Tflipseg + [backflipAngleModel]
        # Model for resting posture
        handsTermTarget = dict()
        qref = copy.deepcopy(self.qref)
        qref[:2] = (LF_pose.translation + RF_pose.translation)[:2] / 2
        pinocchio.forwardKinematics(self.robot_model, self.robot_data, qref)
        for name in [self.LH_name, self.RH_name]:
            frame_id = self.robot_model.getFrameId(name)
            pinocchio.updateFramePlacement(self.robot_model, self.robot_data, frame_id)
            handsTermTarget[name] = self.robot_data.oMf[frame_id]
        bodiesTarget = dict()
        bodiesTarget["torso_1_joint"] = np.eye(3)
        bodiesTarget["torso_2_joint"] = pinocchio.SE3(np.eye(3), qref[:3])
        models += [standModel] * Tend
        terminalModel = self.createModel(
            bodiesTarget=bodiesTarget,
            footContacts=[self.LF_name, self.RF_name],
            handsTarget=handsTermTarget,
        )
        return crocoddyl.ShootingProblem(self.x0, models, terminalModel)

    def _createStateRegCost(self, qref, weights, nu):
        xref = (
            self.xref
            if qref is None
            else np.concatenate([qref, np.zeros(self.state.nv)])
        )
        activation = crocoddyl.ActivationModelWeightedQuad(weights**2)
        residual = crocoddyl.ResidualModelState(self.state, xref, nu)
        return crocoddyl.CostModelResidual(self.state, activation, residual)

    def _createEffortRegCost(self, nu):
        residual = crocoddyl.ResidualModelJointEffort(
            self.state, self.actuation, np.zeros(self.actuation.nu), nu, self._fwddyn
        )
        return crocoddyl.CostModelResidual(self.state, residual)

    def _createAccelerationRegCost(self, nu):
        residual = crocoddyl.ResidualModelJointAcceleration(self.state, nu)
        return crocoddyl.CostModelResidual(self.state, residual)

    def _createForceRegCost(self, frameName, nu):
        frame_id = self.robot_model.getFrameId(frameName)
        fref = pinocchio.Force.Zero()
        residual = crocoddyl.ResidualModelContactForce(
            self.state, frame_id, fref, 6, nu, self._fwddyn
        )
        return crocoddyl.CostModelResidual(self.state, residual)

    def _createStateLimsCost(self, nu):
        bounds = crocoddyl.ActivationBounds(self.xlb, self.xub, 1.0)
        activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        residual = crocoddyl.ResidualModelState(self.state, self.xref, nu)
        return crocoddyl.CostModelResidual(self.state, activation, residual)

    def _createFrictionConeCost(self, frameName, coneRotation, frictionCoeff, nu):
        frame_id = self.robot_model.getFrameId(frameName)
        cone = crocoddyl.FrictionCone(coneRotation, frictionCoeff)
        bounds = crocoddyl.ActivationBounds(cone.lb, cone.ub)
        activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        residual = crocoddyl.ResidualModelContactFrictionCone(
            self.state, frame_id, cone, nu, self._fwddyn
        )
        return crocoddyl.CostModelResidual(self.state, activation, residual)

    def _createWrenchConeCost(
        self, frameName, coneRotation, frictionCoeff, comArea, nu
    ):
        frame_id = self.robot_model.getFrameId(frameName)
        cone = crocoddyl.WrenchCone(coneRotation, frictionCoeff, comArea)
        coneResidual = crocoddyl.ResidualModelContactWrenchCone(
            self.state, frame_id, cone, nu, self._fwddyn
        )
        coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(cone.lb, cone.ub)
        )
        return crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)

    def _createFramePlacementCost(self, frameName, Mref, weights, nu):
        frame_id = self.robot_model.getFrameId(frameName)
        activation = crocoddyl.ActivationModelWeightedQuad(weights**2)
        residual = crocoddyl.ResidualModelFramePlacement(self.state, frame_id, Mref, nu)
        return crocoddyl.CostModelResidual(self.state, activation, residual)

    def _createFrameRotationCost(self, frameName, Rref, weights, nu):
        frame_id = self.robot_model.getFrameId(frameName)
        activation = crocoddyl.ActivationModelWeightedQuad(weights**2)
        residual = crocoddyl.ResidualModelFrameRotation(self.state, frame_id, Rref, nu)
        return crocoddyl.CostModelResidual(self.state, activation, residual)
