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
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
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

        # Defining default state
        q0 = self.rmodel.referenceConfigurations["standing"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

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
        comForwardTermModel.differential.costs.costs["comTrack"].weight = 1e6
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
        comBackwardTermModel.differential.costs.costs["comTrack"].weight = 1e6
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
        comForwardTermModel.differential.costs.costs["comTrack"].weight = 1e6
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
        )
        loco3dModel += doubleSupport + rhStep + rfStep
        loco3dModel += doubleSupport + lhStep + lfStep + [doubleSupport[0]]
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

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
        :return: list of action models for the swing phase and the switch.
        """
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
                )
            ]
        # Action model for the foot switch
        footSwitchModel = self.createSwitch(
            swingFootNames, swingFootTask, constraint=constraint
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
    ):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model.
        :param footContacts: names of the constrained feet.
        :param comTask: optional CoM translation target.
        :param swingFootTask: optional list of [frameName, SE3 target] pairs for
            each swing foot.
        :param constraint: if True, treat friction cones and swing tasks as
            constraints instead of costs.
        :return: integrated action model for the swing phase.
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(footContacts)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ContactModel3D(
                self.state,
                frame_id,
                np.array([0.0, 0.0, 0.0]),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(name + "_contact", supportContactModel)
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        constraintModel = crocoddyl.ConstraintModelManager(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e6)
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
                costModel.addCost(name + "_frictionCone", frictionCone, 1e1)
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
                    costModel.addCost(frame_name + "_footTrack", footTrack, 1e6)
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
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)
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
            costModel.addCost("stateBounds", stateBounds, 1e3)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state,
                self.actuation,
                contactModel,
                costModel,
                constraintModel,
                0.0,
                True,
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel, constraintModel
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
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.four, timeStep
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.three, timeStep
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.two, timeStep
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        return model

    def createSwitch(
        self, footContacts, swingFootTask, pseudoImpulse=False, constraint=False
    ):
        """Action model for a foot switch phase.

        :param footContacts: names of the constrained feet.
        :param swingFootTask: swing foot frame names and landing poses.
        :param pseudoImpulse: True to use pseudo-impulse (cost-based) model,
            False to use impulse dynamics.
        :param constraint: if True, treat swing tasks/friction cones as
            constraints where applicable.
        :return: action model for the foot switch phase.
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(
                footContacts, swingFootTask, constraint
            )
        else:
            return self.createImpulseModel(footContacts, swingFootTask, constraint)

    def createPseudoImpulseModel(self, footContacts, swingFootTask, constraint):
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param footContacts: names of the constrained feet.
        :param swingFootTask: swing foot frame names and landing poses.
        :param constraint: if True, treat swing tasks/friction cones as
            constraints.
        :return: pseudo-impulse differential action model.
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(footContacts)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ContactModel3D(
                self.state,
                frame_id,
                np.array([0.0, 0.0, 0.0]),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(name + "_contact", supportContactModel)
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
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
                costModel.addCost(name + "_frictionCone", frictionCone, 1e1)
            else:
                frictionCone = crocoddyl.ConstraintModelResidual(
                    self.state, coneResidual, cone.lb, cone.ub
                )
                constraintModel.addCost(name + "_frictionCone", frictionCone)
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
                    costModel.addCost(frame_name + "_footTrack", footTrack, 1e7)
                    costModel.addCost(
                        frame_name + "_impulseVel", impulseFootVelCost, 1e6
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
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.four, 0.0
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.three, 0.0
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.two, 0.0)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
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
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ImpulseModel3D(
                self.state, frame_id, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(name + "_impulse", supportContactModel)
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
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulseModel, costModel, constraintModel
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model


def plotSolution(solver, bounds=True, figIndex=1, figTitle="", show=True):
    """Plot joint trajectories, torques and CoM plane for a solver or a list."""
    import matplotlib.pyplot as plt

    xs, us, cs = [], [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []

    def updateTrajectories(solver):
        xs.extend(solver.xs[:-1])
        for m, d in zip(solver.problem.runningModels, solver.problem.runningDatas):
            if hasattr(m, "differential"):
                cs.append(d.differential.multibody.pinocchio.com[0])
                us.append(d.differential.multibody.joint.tau)
                if bounds and isinstance(
                    m.differential, crocoddyl.DifferentialActionModelContactFwdDynamics
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
                solver[0].problem.runningModels[0].differential.actuation.nu,
            )
            updateTrajectories(s)
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        nq, nv, nu = (
            rmodel.nq,
            rmodel.nv,
            solver.problem.runningModels[0].differential.actuation.nu,
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
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 3))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(0, 3))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(0, 3))]
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
