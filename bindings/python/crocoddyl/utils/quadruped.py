# ref. https://github.com/PyCQA/pycodestyle/issues/373, remove this for ruff
import numpy as np
import pinocchio

import crocoddyl


class SimpleQuadrupedalGaitProblem:
    """Helper for assembling simple quadrupedal locomotion problems.

    The class bundles a few canned scenarios used in Crocoddyl examples: walking,
    trotting, pacing, bounding, jumping and small CoM motions. The models are kept
    intentionally simple and are **not** meant for real robots or applications
    beyond tutorial or benchmarking purposes. This file is not part of the public
    API and can change without deprecation.
    """

    def __init__(
        self,
        rmodel,
        lfFoot,
        rfFoot,
        lhFoot,
        rhFoot,
        integrator="euler",
        control="zero",
        fwddyn=True,
        termConstraint=False,
        timeopt=False,
        n_phases=1,
        time_reg_weight=3e-1,
        friction_type=None,
        friction_parameters=None,
    ):
        """Construct quadrupedal-gait problem.

        :param rmodel: Pinocchio robot model used to build states and costs.
        :param lfFoot: name of the left-front foot frame in the model.
        :param rfFoot: name of the right-front foot frame in the model.
        :param lhFoot: name of the left-hind foot frame in the model.
        :param rhFoot: name of the right-hind foot frame in the model.
        :param integrator: discrete integrator for the differential models
            (``"euler"``, ``"rk2"``, ``"rk3"``, ``"rk4"``).
        :param control: control parametrization (``"zero"``, ``"one"``, ``"rk3"``,
            ``"rk4"``); see Crocoddyl control parametrizations for details.
        :param fwddyn: True for forward-dynamics, False for inverse-dynamics
            formulations.
        :param termConstraint: if True, soften running tracking costs and move
            foot tracking to terminal equality constraints where supported.
        :param timeopt: if True, build a parameterized ``ShootingProblem`` with
            one log-time parameter per phase.
        :param n_phases: number of shared time-optimization phases.
        :param time_reg_weight: running weight for the log-time regularization.
        :param friction_type: per-joint friction model used in the actuation
            model. By default, use identity joint actuation.
        :param friction_parameters: log-parameters for the selected friction
            model. When omitted, use nominal Coulomb-viscous parameters.
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.friction_type = friction_type
        self.friction_parameters = (
            np.array([np.log(0.15), np.log(10.0), np.log(0.2)])
            if friction_parameters is None
            else np.asarray(friction_parameters, dtype=float)
        )
        self.actuation = self._createActuationModel()
        self.lfFoot = lfFoot
        self.rfFoot = rfFoot
        self.lhFoot = lhFoot
        self.rhFoot = rhFoot
        # Getting the frame id for all the legs
        self.lfFootId = self.rmodel.getFrameId(lfFoot)
        self.rfFootId = self.rmodel.getFrameId(rfFoot)
        self.lhFootId = self.rmodel.getFrameId(lhFoot)
        self.rhFootId = self.rmodel.getFrameId(rhFoot)
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn
        self._termConstraint = termConstraint
        self._timeopt = timeopt
        self._nphases = n_phases
        self._time_reg_weight = time_reg_weight

        # Defining default state
        q0 = self.rmodel.referenceConfigurations["standing"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

    def _createActuationModel(self):
        if self.friction_type is None:
            return crocoddyl.ActuationModelMultibody(self.state)

        first_actuated_joint = (
            self.rmodel.getJointId("root_joint") + 1
            if self.rmodel.existJointName("root_joint")
            else 1
        )
        joint_dynamics = []
        for jid in range(first_actuated_joint, self.rmodel.njoints):
            joint = self.rmodel.joints[jid]
            if joint.nv == 1:
                joint_dynamics.append(
                    crocoddyl.JointDynamicsModelFriction(
                        jid,
                        joint.nq,
                        self.friction_parameters,
                        self.friction_type,
                    )
                )
            else:
                joint_dynamics.append(
                    crocoddyl.JointDynamicsModelIdentity(jid, joint.nq, joint.nv)
                )
        return crocoddyl.ActuationModelMultibody(self.state, joint_dynamics)

    def _createProblem(self, x0, models, phase_idx=None, phase_times=None):
        if not self._timeopt:
            return crocoddyl.ShootingProblem(x0, models[:-1], models[-1])

        if phase_idx is None:
            phase_idx = [(i * self._nphases) // len(models) for i in range(len(models))]
        phases = [[] for _ in range(self._nphases)]
        for i, model in enumerate(models[:-1]):
            phases[phase_idx[i]].append(model)
        if phase_times is None:
            phase_times = []
            for i, phase in enumerate(phases):
                if not phase:
                    raise ValueError(f"time-opt phase {i} has no running models")
                if any(model is not phase[0] for model in phase):
                    raise ValueError(
                        "phase_times is required when a time-opt phase "
                        "contains multiple model instances"
                    )
                phase_times.append(phase[0].integrator_time)
        if len(phase_times) != self._nphases:
            raise ValueError("phase_times should contain one IntegratorTime per phase")
        params_models = []
        for i, phase in enumerate(phases):
            if not phase:
                raise ValueError(f"time-opt phase {i} has no running models")
            params_i = crocoddyl.ParameterManager(self.state)
            params_i.addParam(
                "timeopt",
                crocoddyl.IntegratorTimeoptParams(self.state, phase_times[i]),
            )
            params_models.append(crocoddyl.ParameterPhaseModel(params_i))
        return crocoddyl.ShootingProblem(x0, phases, models[-1], params_models)

    def _createTerminalFootConstraints(self, models):
        terminal_model = models[-1]
        terminalConstraints = crocoddyl.ConstraintModelManager(
            self.state, terminal_model.nu
        )
        foot_residuals = {}
        for model in models:
            if hasattr(model, "dynamics"):
                for name, item in model.costs.costs.todict().items():
                    if "_footTrack" in name:
                        foot_residuals[name] = item.cost.residual
        for name, residual in foot_residuals.items():
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                self.state, residual.id, residual.reference, terminal_model.nu
            )
            terminalConstraints.addConstraint(
                name,
                crocoddyl.ConstraintModelResidual(self.state, frameTranslationResidual),
            )
        return crocoddyl.IntegratedActionModelEuler(
            terminal_model.dynamics,
            terminal_model.costs,
            terminalConstraints,
            None,
            terminal_model.integrator_time,
        )

    def createCoMProblem(self, x0, comGoTo, timeStep, numKnots, constraint=False):
        """Create a shooting problem for a CoM forward/backward task.

        :param x0: initial state.
        :param comGoTo: forward displacement of the CoM before returning.
        :param timeStep: duration of each node.
        :param numKnots: number of nodes for the forward and backward phases.
        :param constraint: if True, enforce contact constraints instead of
            penalizing them.
        :return: a ``crocoddyl.ShootingProblem`` with forward and backward CoM
            excursions.
        """
        # Compute the current foot positions
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        # Creating the action model for the CoM task
        comModels = []
        comForwardModels = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
            )
            for _ in range(numKnots)
        ]
        comForwardTermModel = self.createModel(
            timeStep=timeStep,
            footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
            comTask=com0 + np.array([comGoTo, 0.0, 0.0]),
            constraint=constraint,
        )
        comForwardTermModel.costs.costs["comTrack"].weight = 1e6
        comBackwardModels = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
            )
            for _ in range(numKnots)
        ]
        comBackwardTermModel = self.createModel(
            timeStep=timeStep,
            footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
            comTask=com0 + np.array([-comGoTo, 0.0, 0.0]),
            constraint=constraint,
        )
        comBackwardTermModel.costs.costs["comTrack"].weight = 1e6
        # Adding the CoM tasks
        comModels += [*comForwardModels, comForwardTermModel]
        comModels += [*comBackwardModels, comBackwardTermModel]
        return crocoddyl.ShootingProblem(x0, comModels[:-1], comModels[-1])

    def createCoMGoalProblem(self, x0, comGoTo, timeStep, numKnots, constraint=False):
        """Create a shooting problem for a CoM position goal task.

        :param x0: initial state.
        :param comGoTo: desired CoM displacement along +x.
        :param timeStep: duration of each node.
        :param numKnots: number of nodes before the terminal knot.
        :param constraint: if True, model friction cones and swing tasks as
            constraints.
        :return: a ``crocoddyl.ShootingProblem`` that reaches the CoM target.
        """
        # Compute the current foot positions
        q0 = x0[: self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        # Creating the action model for the CoM task
        comModels = []
        comForwardModels = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
            )
            for _ in range(numKnots)
        ]
        comForwardTermModel = self.createModel(
            timeStep=timeStep,
            footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
            comTask=com0 + np.array([comGoTo, 0.0, 0.0]),
        )
        comForwardTermModel.costs.costs["comTrack"].weight = 1e6
        # Adding the CoM tasks
        comModels += [*comForwardModels, comForwardTermModel]
        return crocoddyl.ShootingProblem(x0, comModels[:-1], comModels[-1])

    def createWalkingProblem(
        self,
        x0,
        stepLength,
        stepHeight,
        timeStep,
        stepKnots,
        supportKnots,
        constraint=False,
    ):
        """Create a shooting problem for a simple walking gait.

        :param x0: initial state.
        :param stepLength: forward displacement of each footstep.
        :param stepHeight: clearance height during swing.
        :param timeStep: duration of each node.
        :param stepKnots: nodes per swing phase.
        :param supportKnots: nodes for each double-support phase.
        :param constraint: if True, enforce friction cones and foot tracks as
            constraints.
        :return: configured ``crocoddyl.ShootingProblem``.
        """
        # Compute the current foot positions
        q0 = x0[: self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        # Defining the action models along the time instances
        stepKnots_ = stepKnots + 1
        total_knots = (2 * supportKnots) + (4 * stepKnots_) + 1
        phase_idx = [(i * self._nphases) // total_knots for i in range(total_knots)]
        phase_times = (
            [crocoddyl.IntegratorTime(timeStep, True) for _ in range(self._nphases)]
            if self._timeopt
            else None
        )

        def phase_time(knot):
            return phase_times[phase_idx[knot]] if phase_times is not None else None

        loco3dModel = []
        doubleSupportA = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
                integratorTime=phase_time(k),
            )
            for k in range(supportKnots)
        ]
        doubleSupportB = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
                integratorTime=phase_time(supportKnots + 2 * stepKnots_ + k),
            )
            for k in range(supportKnots)
        ]
        if self.firstStep is True:
            rhStep = self.createFootstepModels(
                comRef,
                [rhFootPos0],
                0.5 * stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFoot, self.rfFoot, self.lhFoot],
                [self.rhFoot],
                constraint=constraint,
                integratorTimes=[
                    phase_times[p]
                    for p in phase_idx[supportKnots : supportKnots + stepKnots_]
                ]
                if phase_times is not None
                else None,
            )
            rfStep = self.createFootstepModels(
                comRef,
                [rfFootPos0],
                0.5 * stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFoot, self.lhFoot, self.rhFoot],
                [self.rfFoot],
                constraint=constraint,
                integratorTimes=[
                    phase_times[p]
                    for p in phase_idx[
                        supportKnots + stepKnots_ : supportKnots + 2 * stepKnots_
                    ]
                ]
                if phase_times is not None
                else None,
            )
            self.firstStep = False
        else:
            rhStep = self.createFootstepModels(
                comRef,
                [rhFootPos0],
                stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFoot, self.rfFoot, self.lhFoot],
                [self.rhFoot],
                constraint=constraint,
                integratorTimes=[
                    phase_times[p]
                    for p in phase_idx[supportKnots : supportKnots + stepKnots_]
                ]
                if phase_times is not None
                else None,
            )
            rfStep = self.createFootstepModels(
                comRef,
                [rfFootPos0],
                stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFoot, self.lhFoot, self.rhFoot],
                [self.rfFoot],
                constraint=constraint,
                integratorTimes=[
                    phase_times[p]
                    for p in phase_idx[
                        supportKnots + stepKnots_ : supportKnots + 2 * stepKnots_
                    ]
                ]
                if phase_times is not None
                else None,
            )
        lhStep = self.createFootstepModels(
            comRef,
            [lhFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.lfFoot, self.rfFoot, self.rhFoot],
            [self.lhFoot],
            constraint=constraint,
            integratorTimes=[
                phase_times[p]
                for p in phase_idx[
                    2 * supportKnots + 2 * stepKnots_ : 2 * supportKnots
                    + 3 * stepKnots_
                ]
            ]
            if phase_times is not None
            else None,
        )
        lfStep = self.createFootstepModels(
            comRef,
            [lfFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.rfFoot, self.lhFoot, self.rhFoot],
            [self.lfFoot],
            constraint=constraint,
            integratorTimes=[
                phase_times[p]
                for p in phase_idx[
                    2 * supportKnots + 3 * stepKnots_ : 2 * supportKnots
                    + 4 * stepKnots_
                ]
            ]
            if phase_times is not None
            else None,
        )
        finalDoubleSupport = self.createModel(
            timeStep=timeStep,
            footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
            constraint=constraint,
            integratorTime=phase_time(total_knots - 1),
        )
        loco3dModel += doubleSupportA + rhStep + rfStep
        loco3dModel += doubleSupportB + lhStep + lfStep + [finalDoubleSupport]
        if self._termConstraint:
            loco3dModel[-1] = self._createTerminalFootConstraints(loco3dModel)
        return self._createProblem(x0, loco3dModel, phase_idx, phase_times)

    def createTrottingProblem(
        self,
        x0,
        stepLength,
        stepHeight,
        timeStep,
        stepKnots,
        supportKnots,
        constraint=False,
    ):
        """Create a shooting problem for a simple trotting gait.

        :param x0: initial state.
        :param stepLength: forward displacement of each footstep pair.
        :param stepHeight: clearance height during swing.
        :param timeStep: duration of each node.
        :param stepKnots: nodes per swing phase.
        :param supportKnots: nodes for each double-support phase.
        :param constraint: if True, enforce friction cones and foot tracks as
            constraints.
        :return: configured ``crocoddyl.ShootingProblem``.
        """
        # Compute the current foot positions
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
            )
            for _ in range(supportKnots)
        ]
        if self.firstStep is True:
            rflhStep = self.createFootstepModels(
                comRef,
                [rfFootPos0, lhFootPos0],
                0.5 * stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFoot, self.rhFoot],
                [self.rfFoot, self.lhFoot],
                constraint=constraint,
            )
            self.firstStep = False
        else:
            rflhStep = self.createFootstepModels(
                comRef,
                [rfFootPos0, lhFootPos0],
                stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFoot, self.rhFoot],
                [self.rfFoot, self.lhFoot],
                constraint=constraint,
            )
        lfrhStep = self.createFootstepModels(
            comRef,
            [lfFootPos0, rhFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.rfFoot, self.lhFoot],
            [self.lfFoot, self.rhFoot],
            constraint=constraint,
        )
        loco3dModel += doubleSupport + rflhStep
        loco3dModel += doubleSupport + lfrhStep + [doubleSupport[0]]
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createPacingProblem(
        self,
        x0,
        stepLength,
        stepHeight,
        timeStep,
        stepKnots,
        supportKnots,
        constraint=False,
    ):
        """Create a shooting problem for a simple pacing gait.

        :param x0: initial state.
        :param stepLength: forward displacement of each footstep pair.
        :param stepHeight: clearance height during swing.
        :param timeStep: duration of each node.
        :param stepKnots: nodes per swing phase.
        :param supportKnots: nodes for each double-support phase.
        :param constraint: if True, enforce friction cones and foot tracks as
            constraints.
        :return: configured ``crocoddyl.ShootingProblem``.
        """
        # Compute the current foot positions
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
            )
            for _ in range(supportKnots)
        ]
        if self.firstStep is True:
            rightSteps = self.createFootstepModels(
                comRef,
                [rfFootPos0, rhFootPos0],
                0.5 * stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFoot, self.lhFoot],
                [self.rfFoot, self.rhFoot],
                constraint=constraint,
            )
            self.firstStep = False
        else:
            rightSteps = self.createFootstepModels(
                comRef,
                [rfFootPos0, rhFootPos0],
                stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFoot, self.lhFoot],
                [self.rfFoot, self.rhFoot],
                constraint=constraint,
            )
        leftSteps = self.createFootstepModels(
            comRef,
            [lfFootPos0, lhFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.rfFoot, self.rhFoot],
            [self.lfFoot, self.lhFoot],
            constraint=constraint,
        )
        loco3dModel += doubleSupport + rightSteps
        loco3dModel += doubleSupport + leftSteps + [doubleSupport[0]]
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createBoundingProblem(
        self,
        x0,
        stepLength,
        stepHeight,
        timeStep,
        stepKnots,
        supportKnots,
        constraint=False,
    ):
        """Create a shooting problem for a simple bounding gait.

        :param x0: initial state.
        :param stepLength: forward displacement of each front/hind pair.
        :param stepHeight: clearance height during swing.
        :param timeStep: duration of each node.
        :param stepKnots: nodes per swing phase.
        :param supportKnots: nodes for each double-support phase.
        :param constraint: if True, enforce friction cones and foot tracks as
            constraints.
        :return: configured ``crocoddyl.ShootingProblem``.
        """
        # Compute the current foot positions
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
            )
            for _ in range(supportKnots)
        ]
        hindSteps = self.createFootstepModels(
            comRef,
            [lfFootPos0, rfFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.lhFoot, self.rhFoot],
            [self.lfFoot, self.rfFoot],
            constraint=constraint,
        )
        frontSteps = self.createFootstepModels(
            comRef,
            [lhFootPos0, rhFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.lfFoot, self.rfFoot],
            [self.lhFoot, self.rhFoot],
            constraint=constraint,
        )
        loco3dModel += doubleSupport + hindSteps
        loco3dModel += doubleSupport + frontSteps + [doubleSupport[0]]
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createJumpingProblem(
        self,
        x0,
        jumpHeight,
        jumpLength,
        timeStep,
        groundKnots,
        flyingKnots,
        constraint=False,
    ):
        """Create a shooting problem for a fixed-length jump.

        The sequence follows: crouch/take-off, ballistic flight (up then down),
        touchdown with an impulse/pseudo-impulse phase, and stabilization on the
        landing position.

        :param x0: initial state.
        :param jumpHeight: desired apex height above the initial foot height.
        :param jumpLength: 3D displacement applied to every foot at landing.
        :param timeStep: duration of each node.
        :param groundKnots: nodes during take-off and landing ground phases.
        :param flyingKnots: nodes during the up and down flying phases.
        :param constraint: if True, enforce friction cones and swing tasks as
            constraints.
        :return: configured ``crocoddyl.ShootingProblem``.
        """
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.0
        rhFootPos0[2] = 0.0
        lfFootPos0[2] = 0.0
        lhFootPos0[2] = 0.0
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        # Create locomotion problem
        loco3dModel = []
        takeOff = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                constraint=constraint,
            )
            for _ in range(groundKnots)
        ]
        flyingUpPhase = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[],
                comTask=np.array(
                    [
                        jumpLength[0] / 2.0,
                        jumpLength[1] / 2.0,
                        jumpLength[2] / 2.0 + jumpHeight,
                    ]
                )
                * (k + 1)
                / flyingKnots
                + comRef,
                constraint=constraint,
            )
            for k in range(flyingKnots)
        ]
        flyingDownPhase = []
        for _ in range(flyingKnots):
            flyingDownPhase += [
                self.createModel(
                    timeStep=timeStep, footContacts=[], constraint=constraint
                )
            ]
        f0 = jumpLength
        footTask = [
            [self.lfFoot, pinocchio.SE3(np.eye(3), lfFootPos0 + f0)],
            [self.rfFoot, pinocchio.SE3(np.eye(3), rfFootPos0 + f0)],
            [self.lhFoot, pinocchio.SE3(np.eye(3), lhFootPos0 + f0)],
            [self.rhFoot, pinocchio.SE3(np.eye(3), rhFootPos0 + f0)],
        ]
        landingPhase = [
            self.createSwitch(
                [self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                footTask,
                False,
                constraint=constraint,
            )
        ]
        f0[2] = df
        landed = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot],
                comTask=comRef + f0,
                constraint=constraint,
            )
            for _ in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createFootstepModels(
        self,
        comPos0,
        feetPos0,
        stepLength,
        stepHeight,
        timeStep,
        numKnots,
        footContacts,
        swingFootNames,
        constraint=False,
        integratorTimes=None,
    ):
        """Action models for a footstep phase.

        :param comPos0: initial CoM position.
        :param feetPos0: initial positions of the swinging feet.
        :param stepLength: forward displacement of the swing feet.
        :param stepHeight: clearance height during swing.
        :param timeStep: duration of each node.
        :param numKnots: number of nodes for the swing phase.
        :param footContacts: names of the supporting feet.
        :param swingFootNames: names of the swinging feet.
        :param constraint: if True, enforce friction cones and swing tracks as
            constraints.
        :param integratorTimes: optional list of shared integrator times, one
            per swing knot plus one for the switch.
        :return: list of action models for the swing phase and the switch.
        """
        if integratorTimes is not None and len(integratorTimes) != numKnots + 1:
            raise ValueError("integratorTimes should contain numKnots + 1 entries")
        numLegs = len(footContacts) + len(swingFootNames)
        comPercentage = float(len(swingFootNames)) / numLegs
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for name, p in zip(swingFootNames, feetPos0):
                # Defining a foot swing task given the step length
                # resKnot = numKnots % 2
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0.0, stepHeight * k / phKnots]
                    )
                elif k == phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0.0, stepHeight])
                else:
                    dp = np.array(
                        [
                            stepLength * (k + 1) / numKnots,
                            0.0,
                            stepHeight * (1 - float(k - phKnots) / phKnots),
                        ]
                    )
                tref = p + dp
                swingFootTask += [[name, pinocchio.SE3(np.eye(3), tref)]]
            comTask = (
                np.array([stepLength * (k + 1) / numKnots, 0.0, 0.0]) * comPercentage
                + comPos0
            )
            footSwingModel += [
                self.createModel(
                    timeStep=timeStep,
                    footContacts=footContacts,
                    comTask=comTask,
                    swingFootTask=swingFootTask,
                    constraint=constraint,
                    integratorTime=integratorTimes[k]
                    if integratorTimes is not None
                    else None,
                )
            ]
        # Action model for the foot switch
        footSwitchModel = self.createSwitch(
            swingFootNames,
            swingFootTask,
            constraint=constraint,
            timeStep=timeStep,
            integratorTime=integratorTimes[-1] if integratorTimes is not None else None,
        )
        # Updating the current foot position for next step
        comPos0 += [stepLength * comPercentage, 0.0, 0.0]
        for p in feetPos0:
            p += [stepLength, 0.0, 0.0]
        return [*footSwingModel, footSwitchModel]

    def createModel(
        self,
        timeStep,
        footContacts,
        comTask=None,
        swingFootTask=None,
        constraint=False,
        integratorTime=None,
    ):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model.
        :param footContacts: names of the constrained feet.
        :param comTask: optional CoM translation target.
        :param swingFootTask: optional list of [frameName, SE3 target] pairs for
            each swing foot.
        :param constraint: if True, treat friction cones and swing tasks as
            constraints instead of costs.
        :param integratorTime: optional shared integrator time description.
        :return: integrated action model for the swing phase.
        """
        if integratorTime is None:
            integratorTime = crocoddyl.IntegratorTime(timeStep, self._timeopt)
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(footContacts)
        contactConstraints = crocoddyl.ImplicitConstraintModelMultiple(self.state, nu)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ContactModel(
                self.state,
                frame_id,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
                [True, True, True, False, False, False],
            )
            contactConstraints.addConstraint(name + "_contact", supportContactModel)
        # Creating the cost model for a contact phase
        np_total = 1 if self._timeopt else 0
        costModel = crocoddyl.CostModelSum(self.state, nu, np_total)
        constraintModel = crocoddyl.ConstraintModelManager(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost(
                "comTrack", comTrack, 1e-2 if self._termConstraint else 1e6
            )
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, frame_id, cone, nu, self._fwddyn
            )
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            if not constraint:
                frictionCone = crocoddyl.CostModelResidual(
                    self.state, coneActivation, coneResidual
                )
                costModel.addCost(
                    name + "_frictionCone",
                    frictionCone,
                    1e-2 if self._termConstraint else 1e1,
                )
            else:
                frictionCone = crocoddyl.ConstraintModelResidual(
                    self.state, coneResidual, cone.lb, cone.ub
                )
                constraintModel.addConstraint(name + "_frictionCone", frictionCone)
        if swingFootTask is not None:
            for target in swingFootTask:
                frame_name, placement = target
                frame_id = self.rmodel.getFrameId(frame_name)
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, frame_id, placement.translation, nu
                )
                if True:  # not constraint: TODO: evaluate this further with restoring mechanism
                    footTrack = crocoddyl.CostModelResidual(
                        self.state, frameTranslationResidual
                    )
                    costModel.addCost(
                        frame_name + "_footTrack",
                        footTrack,
                        1e-1 if self._termConstraint else 1e6,
                    )
                else:
                    footTrack = crocoddyl.ConstraintModelResidual(
                        self.state, frameTranslationResidual
                    )
                    constraintModel.addConstraint(frame_name + "_footTrack", footTrack)
        stateWeights = np.array(
            [0.0] * 3
            + [500.0] * 3
            + [0.01] * (self.rmodel.nv - 6)
            + [10.0] * 6
            + [1.0] * (self.rmodel.nv - 6)
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        if self._timeopt:
            pResidual = crocoddyl.ResidualModelParameters(
                self.state, np.array([np.log(integratorTime.timeStep)]), nu
            )
            pRegCost = crocoddyl.CostModelResidual(self.state, pResidual)
            costModel.addCost("pReg", pRegCost, self._time_reg_weight)
        costModel.addCost("stateReg", stateReg, 1e-4 if self._termConstraint else 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-4 if self._termConstraint else 1e-1)
        lb = np.concatenate(
            [self.state.lb[1 : self.state.nv + 1], self.state.lb[-self.state.nv :]]
        )
        ub = np.concatenate(
            [self.state.ub[1 : self.state.nv + 1], self.state.ub[-self.state.nv :]]
        )
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        if not constraint:
            stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(lb, ub)
            )
            stateBounds = crocoddyl.CostModelResidual(
                self.state, stateBoundsActivation, stateBoundsResidual
            )
            costModel.addCost(
                "stateBounds", stateBounds, 1e-3 if self._termConstraint else 1e3
            )
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dynamics = crocoddyl.DynamicsModelConstrainedForward(
                self.state, self.actuation, contactConstraints, np_total
            )
        else:
            dynamics = crocoddyl.DynamicsModelConstrainedInverse(
                self.state, self.actuation, contactConstraints, np_total
            )
        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.four
            )
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.three
            )
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(
                dynamics,
                costModel,
                constraintModel,
                control,
                integratorTime,
            )
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dynamics,
                costModel,
                constraintModel,
                control,
                integratorTime,
                crocoddyl.RKType.four,
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dynamics,
                costModel,
                constraintModel,
                control,
                integratorTime,
                crocoddyl.RKType.three,
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dynamics,
                costModel,
                constraintModel,
                control,
                integratorTime,
                crocoddyl.RKType.two,
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(
                dynamics,
                costModel,
                constraintModel,
                control,
                integratorTime,
            )
        if self._fwddyn:
            u_lb = np.empty(model.nu)
            u_ub = np.empty(model.nu)
            control.convertBounds(self.actuation.u_lb, self.actuation.u_ub, u_lb, u_ub)
            model.u_lb = u_lb
            model.u_ub = u_ub
        return model

    def createSwitch(
        self,
        footContacts,
        swingFootTask,
        pseudoImpulse=False,
        constraint=False,
        timeStep=0.0,
        integratorTime=None,
    ):
        """Action model for a foot switch phase.

        :param footContacts: names of the constrained feet.
        :param swingFootTask: swing foot frame names and landing poses.
        :param pseudoImpulse: True to use pseudo-impulse (cost-based) model,
            False to use impulse dynamics.
        :param constraint: if True, treat swing tasks/friction cones as
            constraints where applicable.
        :param timeStep: duration of a pseudo-impulse node.
        :param integratorTime: optional shared integrator time description.
        :return: action model for the foot switch phase.
        """
        if pseudoImpulse or self._timeopt:
            return self.createPseudoImpulseModel(
                footContacts,
                swingFootTask,
                constraint,
                timeStep=timeStep,
                integratorTime=integratorTime,
            )
        else:
            return self.createImpulseModel(footContacts, swingFootTask, constraint)

    def createPseudoImpulseModel(
        self,
        footContacts,
        swingFootTask,
        constraint,
        timeStep=0.0,
        integratorTime=None,
    ):
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param footContacts: names of the constrained feet.
        :param swingFootTask: swing foot frame names and landing poses.
        :param constraint: if True, treat swing tasks/friction cones as
            constraints.
        :param timeStep: duration of the pseudo-impulse node.
        :param integratorTime: optional shared integrator time description.
        :return: pseudo-impulse differential action model.
        """
        if integratorTime is None:
            integratorTime = crocoddyl.IntegratorTime(
                timeStep if self._timeopt else 0.0, self._timeopt
            )
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(footContacts)
        contactConstraints = crocoddyl.ImplicitConstraintModelMultiple(self.state, nu)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ContactModel(
                self.state,
                frame_id,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
                [True, True, True, False, False, False],
            )
            contactConstraints.addConstraint(name + "_contact", supportContactModel)
        # Creating the cost model for a contact phase
        np_total = 1 if self._timeopt else 0
        costModel = crocoddyl.CostModelSum(self.state, nu, np_total)
        constraintModel = crocoddyl.ConstraintModelManager(self.state, nu)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, frame_id, cone, nu, self._fwddyn
            )
            if not constraint:
                coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
                    crocoddyl.ActivationBounds(cone.lb, cone.ub)
                )
                frictionCone = crocoddyl.CostModelResidual(
                    self.state, coneActivation, coneResidual
                )
                costModel.addCost(
                    name + "_frictionCone",
                    frictionCone,
                    1e-2 if self._termConstraint else 1e1,
                )
            else:
                frictionCone = crocoddyl.ConstraintModelResidual(
                    self.state, coneResidual, cone.lb, cone.ub
                )
                constraintModel.addConstraint(name + "_frictionCone", frictionCone)
        if swingFootTask is not None:
            for target in swingFootTask:
                frame_name, placement = target
                frame_id = self.rmodel.getFrameId(frame_name)
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, frame_id, placement.translation, nu
                )
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    frame_id,
                    pinocchio.Motion.Zero(),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                if not constraint:
                    footTrack = crocoddyl.CostModelResidual(
                        self.state, frameTranslationResidual
                    )
                    impulseFootVelCost = crocoddyl.CostModelResidual(
                        self.state, frameVelocityResidual
                    )
                    costModel.addCost(
                        frame_name + "_footTrack",
                        footTrack,
                        1.0 if self._termConstraint else 1e7,
                    )
                    costModel.addCost(
                        frame_name + "_impulseVel",
                        impulseFootVelCost,
                        1e-1 if self._termConstraint else 1e6,
                    )
                else:
                    footTrack = crocoddyl.ConstraintModelResidual(
                        self.state, frameTranslationResidual
                    )
                    impulseFootVelCost = crocoddyl.ConstraintModelResidual(
                        self.state, frameVelocityResidual
                    )
                    constraintModel.addConstraint(frame_name + "_footTrack", footTrack)
                    constraintModel.addConstraint(
                        frame_name + "_impulseVel", impulseFootVelCost
                    )
        stateWeights = np.array(
            [0.0] * 3
            + [500.0] * 3
            + [0.01] * (self.rmodel.nv - 6)
            + [10.0] * self.rmodel.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        if self._timeopt:
            pResidual = crocoddyl.ResidualModelParameters(
                self.state, np.array([np.log(integratorTime.timeStep)]), nu
            )
            pRegCost = crocoddyl.CostModelResidual(self.state, pResidual)
            costModel.addCost("pReg", pRegCost, self._time_reg_weight)
        costModel.addCost("stateReg", stateReg, 1e-4 if self._termConstraint else 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-4 if self._termConstraint else 1e-3)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dynamics = crocoddyl.DynamicsModelConstrainedForward(
                self.state, self.actuation, contactConstraints, np_total
            )
        else:
            dynamics = crocoddyl.DynamicsModelConstrainedInverse(
                self.state, self.actuation, contactConstraints, np_total
            )
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(
                dynamics,
                costModel,
                constraintModel,
                None,
                integratorTime,
            )
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dynamics,
                costModel,
                constraintModel,
                None,
                integratorTime,
                crocoddyl.RKType.four,
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dynamics,
                costModel,
                constraintModel,
                None,
                integratorTime,
                crocoddyl.RKType.three,
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dynamics,
                costModel,
                constraintModel,
                None,
                integratorTime,
                crocoddyl.RKType.two,
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(
                dynamics,
                costModel,
                constraintModel,
                None,
                integratorTime,
            )
        if self._fwddyn:
            model.u_lb = self.actuation.u_lb
            model.u_ub = self.actuation.u_ub
        return model

    def createImpulseModel(
        self,
        footContacts,
        swingFootTask,
        JMinvJt_damping=1e-12,
        r_coeff=0.0,
        constraint=False,
    ):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param footContacts: names of the constrained feet.
        :param swingFootTask: swing foot frame names and landing poses.
        :param JMinvJt_damping: damping applied to the impulse dynamics solver.
        :param r_coeff: restitution coefficient for the impulse dynamics.
        :param constraint: if True, treat swing tasks as constraints.
        :return: impulse action model.
        """
        # Creating 3D impulse constraints for the supporting feet
        contactConstraints = crocoddyl.ImplicitConstraintModelMultiple(self.state, 0)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ContactModel(
                self.state,
                frame_id,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                0,
                np.zeros(2),
                [True, True, True, False, False, False],
            )
            contactConstraints.addConstraint(name + "_impulse", supportContactModel)
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        constraintModel = crocoddyl.ConstraintModelManager(self.state, 0)
        if swingFootTask is not None:
            for target in swingFootTask:
                frame_name, placement = target
                frame_id = self.rmodel.getFrameId(frame_name)
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, frame_id, placement.translation, 0
                )
                if not constraint:
                    footTrack = crocoddyl.CostModelResidual(
                        self.state, frameTranslationResidual
                    )
                    costModel.addCost(frame_name + "_footTrack", footTrack, 1e7)
                else:
                    footTrack = crocoddyl.ConstraintModelResidual(
                        self.state, frameTranslationResidual
                    )
                    constraintModel.addConstraint(frame_name + "_footTrack", footTrack)
        stateWeights = np.array(
            [1.0] * 6 + [10.0] * (self.rmodel.nv - 6) + [10.0] * self.rmodel.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, 0
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        dynamics = crocoddyl.DynamicsModelImpulseForward(
            self.state, contactConstraints, 0, r_coeff, JMinvJt_damping
        )
        return crocoddyl.DiscretizedActionModel(dynamics, costModel, constraintModel)


def plotSolution(solver, bounds=True, figIndex=1, figTitle="", show=True):
    """Plot joint trajectories, torques and CoM plane for a solver or a list."""
    import matplotlib.pyplot as plt

    xs, us, cs = [], [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []

    def updateTrajectories(solver):
        xs.extend(solver.xs.tolist()[:-1])
        for m, d in zip(solver.problem.runningModels, solver.problem.runningDatas):
            if hasattr(m, "dynamics"):
                cs.append(d.dynamics.multibody.pinocchio.com[0])
                us.append(d.dynamics.multibody.joint.tau)
                if bounds and isinstance(
                    m.dynamics, crocoddyl.DynamicsModelConstrainedForward
                ):
                    us_lb.extend([m.u_lb])
                    us_ub.extend([m.u_ub])
            else:
                cs.append(d.multibody.pinocchio.com[0])
                us.append(np.zeros(nu))
                if bounds:
                    us_lb.append(np.nan * np.ones(nu))
                    us_ub.append(np.nan * np.ones(nu))
            if bounds:
                xs_lb.extend([m.state.lb])
                xs_ub.extend([m.state.ub])

    if isinstance(solver, list):
        for s in solver:
            rmodel = solver[0].problem.runningModels[0].state.pinocchio
            nq, nv, nu = (
                rmodel.nq,
                rmodel.nv,
                solver[0].problem.runningModels[0].dynamics.actuation.nu,
            )
            updateTrajectories(s)
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        nq, nv, nu = (
            rmodel.nq,
            rmodel.nv,
            solver.problem.runningModels[0].dynamics.actuation.nu,
        )
        updateTrajectories(solver)

    # Getting the state and control trajectories
    nx = nq + nv
    X = [0.0] * nx
    U = [0.0] * nu
    if bounds:
        U_LB = [0.0] * nu
        U_UB = [0.0] * nu
        X_LB = [0.0] * nx
        X_UB = [0.0] * nx
    for i in range(nx):
        X[i] = [x[i] for x in xs]
        if bounds:
            X_LB[i] = [x[i] for x in xs_lb]
            X_UB[i] = [x[i] for x in xs_ub]
    for i in range(nu):
        U[i] = [u[i] for u in us]
        if bounds:
            U_LB[i] = [u[i] for u in us_lb]
            U_UB[i] = [u[i] for u in us_ub]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ["HAA", "HFE", "KFE"]
    # LF foot
    plt.subplot(4, 3, 1)
    plt.title("joint position [rad]")
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 10))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(7, 10))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(7, 10))]
    plt.ylabel("LF")
    plt.legend()
    plt.subplot(4, 3, 2)
    plt.title("joint velocity [rad/s]")
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 6, nq + 9))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 6, nq + 9))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 6, nq + 9))]
    plt.ylabel("LF")
    plt.legend()
    plt.subplot(4, 3, 3)
    plt.title("joint torque [Nm]")
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(3))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(3))]
    plt.ylabel("LF")
    plt.legend()

    # LH foot
    plt.subplot(4, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(10, 13))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(10, 13))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(10, 13))]
    plt.ylabel("LH")
    plt.legend()
    plt.subplot(4, 3, 5)
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 9, nq + 12))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 9, nq + 12))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 9, nq + 12))]
    plt.ylabel("LH")
    plt.legend()
    plt.subplot(4, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3, 6))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(3, 6))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(3, 6))]
    plt.ylabel("LH")
    plt.legend()

    # RF foot
    plt.subplot(4, 3, 7)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 16))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(13, 16))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(13, 16))]
    plt.ylabel("RF")
    plt.legend()
    plt.subplot(4, 3, 8)
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 12, nq + 15))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 12, nq + 15))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 12, nq + 15))]
    plt.ylabel("RF")
    plt.legend()
    plt.subplot(4, 3, 9)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(6, 9))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(6, 9))]
    plt.ylabel("RF")
    plt.legend()

    # RH foot
    plt.subplot(4, 3, 10)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(16, 19))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(16, 19))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(16, 19))]
    plt.ylabel("RH")
    plt.xlabel("knots")
    plt.legend()
    plt.subplot(4, 3, 11)
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 15, nq + 18))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 15, nq + 18))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 15, nq + 18))]
    plt.ylabel("RH")
    plt.xlabel("knots")
    plt.legend()
    plt.subplot(4, 3, 12)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(9, 12))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(9, 12))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(9, 12))]
    plt.ylabel("RH")
    plt.legend()
    plt.xlabel("knots")

    plt.figure(figIndex + 1)
    plt.suptitle(figTitle)
    Cx = [c[0] for c in cs]
    Cy = [c[1] for c in cs]
    plt.plot(Cx, Cy)
    plt.title("CoM position")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    if show:
        plt.show()
