# ref. https://github.com/PyCQA/pycodestyle/issues/373, remove this for ruff
import numpy as np
import pinocchio

import crocoddyl


class SimpleQuadrupedalGaitProblem:
    """Build simple quadrupedal locomotion problems.

    This class aims to build simple locomotion problems used in the examples of
    Crocoddyl.
    The scope of this class is purely for academic reasons, and it does not aim to be
    used in any robotics application.
    We also do not consider it as part of the API, so changes in this class will not
    pass through a strict process of deprecation.
    Thus, we advice any user to DO NOT develop their application based on this class.
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

        :param rmodel: robot model
        :param lfFoot: name of the left-front foot
        :param rfFoot: name of the right-front foot
        :param lhFoot: name of the left-hind foot
        :param rhFoot: name of the right-hind foot
        :param integrator: type of the integrator
            (options are: 'euler', and 'rk4')
        :param control: type of control parametrization
            (options are: 'zero', 'one', and 'rk4')
        :param fwddyn: True for forward-dynamics and False for inverse-dynamics
            formulations
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
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

    def createCoMProblem(self, x0, comGoTo, timeStep, numKnots):
        """Create a shooting problem for a CoM forward/backward task.

        :param x0: initial state
        :param comGoTo: initial CoM motion
        :param timeStep: step time for each knot
        :param numKnots: number of knots per each phase
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        # Creating the action model for the CoM task
        comModels = []
        comForwardModels = [
            self.createSwingFootModel(
                timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
            )
            for _ in range(numKnots)
        ]
        comForwardTermModel = self.createSwingFootModel(
            timeStep,
            [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            com0 + np.array([comGoTo, 0.0, 0.0]),
        )
        comForwardTermModel.differential.costs.costs["comTrack"].weight = 1e6
        comBackwardModels = [
            self.createSwingFootModel(
                timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
            )
            for _ in range(numKnots)
        ]
        comBackwardTermModel = self.createSwingFootModel(
            timeStep,
            [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            com0 + np.array([-comGoTo, 0.0, 0.0]),
        )
        comBackwardTermModel.differential.costs.costs["comTrack"].weight = 1e6
        # Adding the CoM tasks
        comModels += [*comForwardModels, comForwardTermModel]
        comModels += [*comBackwardModels, comBackwardTermModel]
        return crocoddyl.ShootingProblem(x0, comModels[:-1], comModels[-1])

    def createCoMGoalProblem(self, x0, comGoTo, timeStep, numKnots):
        """Create a shooting problem for a CoM position goal task.

        :param x0: initial state
        :param comGoTo: CoM position change target
        :param timeStep: step time for each knot
        :param numKnots: number of knots per each phase
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[: self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        # Creating the action model for the CoM task
        comModels = []
        comForwardModels = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            )
            for _ in range(numKnots)
        ]
        comForwardTermModel = self.createSwingFootModel(
            timeStep,
            [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            com0 + np.array([comGoTo, 0.0, 0.0]),
        )
        comForwardTermModel.differential.costs.costs["comTrack"].weight = 1e6
        # Adding the CoM tasks
        comModels += [*comForwardModels, comForwardTermModel]
        return crocoddyl.ShootingProblem(x0, comModels[:-1], comModels[-1])

    def createWalkingProblem(
        self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
    ):
        """Create a shooting problem for a simple walking gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
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
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
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
                [self.lfFootId, self.rfFootId, self.lhFootId],
                [self.rhFootId],
            )
            rfStep = self.createFootstepModels(
                comRef,
                [rfFootPos0],
                0.5 * stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFootId, self.lhFootId, self.rhFootId],
                [self.rfFootId],
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
                [self.lfFootId, self.rfFootId, self.lhFootId],
                [self.rhFootId],
            )
            rfStep = self.createFootstepModels(
                comRef,
                [rfFootPos0],
                stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.lfFootId, self.lhFootId, self.rhFootId],
                [self.rfFootId],
            )
        lhStep = self.createFootstepModels(
            comRef,
            [lhFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.lfFootId, self.rfFootId, self.rhFootId],
            [self.lhFootId],
        )
        lfStep = self.createFootstepModels(
            comRef,
            [lfFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.rfFootId, self.lhFootId, self.rhFootId],
            [self.lfFootId],
        )
        loco3dModel += doubleSupport + rhStep + rfStep
        loco3dModel += doubleSupport + lhStep + lfStep + [doubleSupport[0]]
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createTrottingProblem(
        self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
    ):
        """Create a shooting problem for a simple trotting gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
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
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
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
                [self.lfFootId, self.rhFootId],
                [self.rfFootId, self.lhFootId],
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
                [self.lfFootId, self.rhFootId],
                [self.rfFootId, self.lhFootId],
            )
        lfrhStep = self.createFootstepModels(
            comRef,
            [lfFootPos0, rhFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.rfFootId, self.lhFootId],
            [self.lfFootId, self.rhFootId],
        )
        loco3dModel += doubleSupport + rflhStep
        loco3dModel += doubleSupport + lfrhStep + [doubleSupport[0]]
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createPacingProblem(
        self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
    ):
        """Create a shooting problem for a simple pacing gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
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
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
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
                [self.lfFootId, self.lhFootId],
                [self.rfFootId, self.rhFootId],
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
                [self.lfFootId, self.lhFootId],
                [self.rfFootId, self.rhFootId],
            )
        leftSteps = self.createFootstepModels(
            comRef,
            [lfFootPos0, lhFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.rfFootId, self.rhFootId],
            [self.lfFootId, self.lhFootId],
        )
        loco3dModel += doubleSupport + rightSteps
        loco3dModel += doubleSupport + leftSteps + [doubleSupport[0]]
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createBoundingProblem(
        self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
    ):
        """Create a shooting problem for a simple bounding gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
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
            self.createSwingFootModel(
                timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
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
            [self.lhFootId, self.rhFootId],
            [self.lfFootId, self.rfFootId],
        )
        frontSteps = self.createFootstepModels(
            comRef,
            [lhFootPos0, rhFootPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.lfFootId, self.rfFootId],
            [self.lhFootId, self.rhFootId],
        )
        loco3dModel += doubleSupport + hindSteps
        loco3dModel += doubleSupport + frontSteps + [doubleSupport[0]]
        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createJumpingProblem(
        self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots
    ):
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
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            )
            for _ in range(groundKnots)
        ]
        flyingUpPhase = [
            self.createSwingFootModel(
                timeStep,
                [],
                np.array(
                    [
                        jumpLength[0] / 2.0,
                        jumpLength[1] / 2.0,
                        jumpLength[2] / 2.0 + jumpHeight,
                    ]
                )
                * (k + 1)
                / flyingKnots
                + comRef,
            )
            for k in range(flyingKnots)
        ]
        flyingDownPhase = []
        for _ in range(flyingKnots):
            flyingDownPhase += [self.createSwingFootModel(timeStep, [])]
        f0 = jumpLength
        footTask = [
            [self.lfFootId, pinocchio.SE3(np.eye(3), lfFootPos0 + f0)],
            [self.rfFootId, pinocchio.SE3(np.eye(3), rfFootPos0 + f0)],
            [self.lhFootId, pinocchio.SE3(np.eye(3), lhFootPos0 + f0)],
            [self.rhFootId, pinocchio.SE3(np.eye(3), rhFootPos0 + f0)],
        ]
        landingPhase = [
            self.createFootSwitchModel(
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                footTask,
                False,
            )
        ]
        f0[2] = df
        landed = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                comTask=comRef + f0,
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
        supportFootIds,
        swingFootIds,
    ):
        """Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
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
                swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]
            comTask = (
                np.array([stepLength * (k + 1) / numKnots, 0.0, 0.0]) * comPercentage
                + comPos0
            )
            footSwingModel += [
                self.createSwingFootModel(
                    timeStep,
                    supportFootIds,
                    comTask=comTask,
                    swingFootTask=swingFootTask,
                )
            ]
        # Action model for the foot switch
        footSwitchModel = self.createFootSwitchModel(swingFootIds, swingFootTask)
        # Updating the current foot position for next step
        comPos0 += [stepLength * comPercentage, 0.0, 0.0]
        for p in feetPos0:
            p += [stepLength, 0.0, 0.0]
        return [*footSwingModel, footSwitchModel]

    def createSwingFootModel(
        self, timeStep, supportFootIds, comTask=None, swingFootTask=None
    ):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(supportFootIds)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(
                self.state,
                i,
                np.array([0.0, 0.0, 0.0]),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e6)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, i, cone, nu, self._fwddyn
            )
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            frictionCone = crocoddyl.CostModelResidual(
                self.state, coneActivation, coneResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1
            )
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i[0], i[1].translation, nu
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, frameTranslationResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6
                )
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
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
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

    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=False):
        """Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the
            impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(supportFootIds)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(
                self.state,
                i,
                np.array([0.0, 0.0, 0.0]),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, i, cone, nu, self._fwddyn
            )
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            frictionCone = crocoddyl.CostModelResidual(
                self.state, coneActivation, coneResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1
            )
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i[0], i[1].translation, nu
                )
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i[0],
                    pinocchio.Motion.Zero(),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, frameTranslationResidual
                )
                impulseFootVelCost = crocoddyl.CostModelResidual(
                    self.state, frameVelocityResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e7
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_impulseVel",
                    impulseFootVelCost,
                    1e6,
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
        self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0
    ):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel3D(
                self.state, i, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(
                self.rmodel.frames[i].name + "_impulse", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i[0], i[1].translation, 0
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, frameTranslationResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e7
                )
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
            self.state, impulseModel, costModel
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model


def plotSolution(solver, bounds=True, figIndex=1, figTitle="", show=True):
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
