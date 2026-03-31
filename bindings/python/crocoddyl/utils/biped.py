import numpy as np
import pinocchio

import crocoddyl


class SimpleBipedGaitProblem:
    """Helper for assembling simple bipedal locomotion problems.

    The class bundles a few canned scenarios used in Crocoddyl examples (walking
    and jumping). The models are intentionally simple and **not** intended for
    real robots or production applications. This file is not part of the public
    API and can change without deprecation.
    """

    def __init__(
        self,
        rmodel,
        rightFoot,
        leftFoot,
        integrator="euler",
        control="zero",
        fwddyn=True,
    ):
        """Construct biped-gait problem.

        :param rmodel: Pinocchio robot model used to build states and costs.
        :param rightFoot: name of the right foot frame in the model.
        :param leftFoot: name of the left foot frame in the model.
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
        self.rightFoot = rightFoot
        self.leftFoot = leftFoot
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

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
        :param constraint: if True, enforce friction cones and swing tracks as
            constraints.
        :return: configured ``crocoddyl.ShootingProblem``.
        """
        # Compute the current foot positions
        q0 = x0[: self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]
        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.rightFoot, self.leftFoot],
                constraint=constraint,
            )
            for _ in range(supportKnots)
        ]
        # Creating the action models for three steps
        if self.firstStep is True:
            rStep = self.createFootstepModels(
                comRef,
                [rfPos0],
                0.5 * stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.leftFoot],
                [self.rightFoot],
                constraint=constraint,
            )
            self.firstStep = False
        else:
            rStep = self.createFootstepModels(
                comRef,
                [rfPos0],
                stepLength,
                stepHeight,
                timeStep,
                stepKnots,
                [self.leftFoot],
                [self.rightFoot],
                constraint=constraint,
            )
        lStep = self.createFootstepModels(
            comRef,
            [lfPos0],
            stepLength,
            stepHeight,
            timeStep,
            stepKnots,
            [self.rightFoot],
            [self.leftFoot],
            constraint=constraint,
        )
        # We defined the problem as:
        loco3dModel += doubleSupport + rStep
        loco3dModel += doubleSupport + lStep
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

        :param x0: initial state.
        :param jumpHeight: desired apex height above the initial foot height.
        :param jumpLength: 3D displacement applied to both feet at landing.
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
        rfFootPos0 = self.rdata.oMf[self.rfId].translation
        lfFootPos0 = self.rdata.oMf[self.lfId].translation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.0
        lfFootPos0[2] = 0.0
        comRef = (rfFootPos0 + lfFootPos0) / 2
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]
        # Create locomotion problem
        loco3dModel = []
        takeOff = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.leftFoot, self.rightFoot],
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
            [self.leftFoot, pinocchio.SE3(np.eye(3), lfFootPos0 + f0)],
            [self.rightFoot, pinocchio.SE3(np.eye(3), rfFootPos0 + f0)],
        ]
        landingPhase = [
            self.createSwitch(
                [self.leftFoot, self.rightFoot], footTask, False, constraint=constraint
            )
        ]
        f0[2] = df
        landed = [
            self.createModel(
                timeStep=timeStep,
                footContacts=[self.leftFoot, self.rightFoot],
                comTask=comRef + f0,
                constraint=constraint,
            )
            for _ in range(int(groundKnots / 2))
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
        :param numKnots: number of nodes for the footstep phase.
        :param footContacts: names of the supporting feet.
        :param swingFootNames: names of the swinging feet.
        :param constraint: if True, enforce friction cones and swing tracks as
            constraints.
        :return: footstep action models.
        """
        numLegs = len(footContacts) + len(swingFootNames)
        comPercentage = float(len(swingFootNames)) / numLegs
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for name, p in zip(swingFootNames, feetPos0):
                # Defining a foot swing task given the step length. The swing task
                # is decomposed on two phases: swing-up and swing-down. We decide
                # deliveratively to allocated the same number of nodes (i.e. phKnots)
                # in each phase. With this, we define a proper z-component for the
                # swing-leg motion.
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
            swingFootNames, swingFootTask, pseudoImpulse=False, constraint=constraint
        )
        # Updating the current foot position for next step
        comPos0 += [stepLength * comPercentage, 0.0, 0.0]
        for p in feetPos0:
            p += [stepLength, 0.0, 0.0]
        return [*footSwingModel, footSwitchModel]

    def createModel(
        self, timeStep, footContacts, comTask=None, swingFootTask=None, constraint=False
    ):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model.
        :param footContacts: names of the constrained feet.
        :param comTask: optional CoM task.
        :param swingFootTask: optional list of [frameName, SE3 target] pairs for
            each swing foot.
        :param constraint: if True, treat friction cones and swing tasks as
            constraints instead of costs.
        :return: action model for a swing foot phase.
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(footContacts)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                frame_id,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 30.0]),
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
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, frame_id, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            if not constraint:
                wrenchCone = crocoddyl.CostModelResidual(
                    self.state, wrenchActivation, wrenchResidual
                )
                costModel.addCost(name + "_wrenchCone", wrenchCone, 1e1)
            else:
                wrenchCone = crocoddyl.ConstraintModelResidual(
                    self.state, wrenchResidual, cone.lb, cone.ub
                )
                constraintModel.addConstraint(
                    name + "_wrenchCone",
                    wrenchCone,
                )
        if swingFootTask is not None:
            for target in swingFootTask:
                frame_name, placement = target
                frame_id = self.rmodel.getFrameId(frame_name)
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, frame_id, placement, nu
                )
                if True:  # not constraint: TODO: evaluate this further with restoring mechanism
                    footTrack = crocoddyl.CostModelResidual(
                        self.state, framePlacementResidual
                    )
                    costModel.addCost(frame_name + "_footTrack", footTrack, 1e6)
                else:
                    footTrack = crocoddyl.ConstraintModelResidual(
                        self.state, framePlacementResidual
                    )
                    constraintModel.addConstraint(
                        frame_name + "_footTrack",
                        footTrack,
                    )
        stateWeights = np.array(
            [0] * 3 + [500.0] * 3 + [0.01] * (self.state.nv - 6) + [10] * self.state.nv
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
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)
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
        :param pseudoImpulse: True for pseudo-impulse models, otherwise impulse.
        :param constraint: if True, treat swing tasks/friction cones as
            constraints where applicable.
        :return: action model for a foot switch phase.
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(
                footContacts, swingFootTask, constraint
            )
        else:
            return self.createImpulseModel(footContacts, swingFootTask, constraint)

    def createPseudoImpulseModel(self, footContacts, swingFootTask, constraint=False):
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param footContacts: names of the constrained feet.
        :param swingFootTask: swing foot frame names and landing poses.
        :param constraint: if True, enforce wrench cone and swing tasks as
            constraints.
        :return: pseudo-impulse differential action model.
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(footContacts)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                frame_id,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(name + "_contact", supportContactModel)
        # Creating the cost/constraint model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        constraintModel = crocoddyl.ConstraintModelManager(self.state, nu)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, frame_id, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            if not constraint:
                wrenchCone = crocoddyl.CostModelResidual(
                    self.state, wrenchActivation, wrenchResidual
                )
                costModel.addCost(name + "_wrenchCone", wrenchCone, 1e1)
            else:
                wrenchCone = crocoddyl.ConstraintModelResidual(
                    self.state, wrenchResidual, cone.lb, cone.ub
                )
                constraintModel.addConstraint(name + "_wrenchCone", wrenchCone)
        if swingFootTask is not None:
            for target in swingFootTask:
                frame_name, placement = target
                frame_id = self.rmodel.getFrameId(frame_name)
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, frame_id, placement, nu
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
                        self.state, framePlacementResidual
                    )
                    impulseFootVelCost = crocoddyl.CostModelResidual(
                        self.state, frameVelocityResidual
                    )
                    costModel.addCost(frame_name + "_footTrack", footTrack, 1e8)
                    costModel.addCost(
                        frame_name + "_impulseVel", impulseFootVelCost, 1e6
                    )
                else:
                    footTrack = crocoddyl.ConstraintModelResidual(
                        self.state, framePlacementResidual
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
            + [0.01] * (self.state.nv - 6)
            + [10] * self.state.nv
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
        :param swingFootTask: swinging foot task.
        :param JMinvJt_damping: damping applied to the impulse dynamics solver.
        :param r_coeff: restitution coefficient for the impulse dynamics.
        :param constraint: if True, treat swing tasks as constraints.
        :return impulse action model
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for name in footContacts:
            frame_id = self.rmodel.getFrameId(name)
            supportContactModel = crocoddyl.ImpulseModel6D(
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
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, frame_id, placement, 0
                )
                if not constraint:
                    footTrack = crocoddyl.CostModelResidual(
                        self.state, framePlacementResidual
                    )
                    costModel.addCost(frame_name + "_footTrack", footTrack, 1e8)
                else:
                    footTrack = crocoddyl.ConstraintModelResidual(
                        self.state, framePlacementResidual
                    )
                    constraintModel.addConstraint(
                        frame_name + "_footTrack",
                        footTrack,
                    )
        stateWeights = np.array(
            [1.0] * 6 + [0.1] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv
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
    legJointNames = ["1", "2", "3", "4", "5", "6"]
    # left foot
    plt.subplot(2, 3, 1)
    plt.title("joint position [rad]")
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 13))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(7, 13))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(7, 13))]
    plt.ylabel("LF")
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.title("joint velocity [rad/s]")
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 6, nq + 12))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 6, nq + 12))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 6, nq + 12))]
    plt.ylabel("LF")
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.title("joint torque [Nm]")
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 6))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(0, 6))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(0, 6))]
    plt.ylabel("LF")
    plt.legend()

    # right foot
    plt.subplot(2, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 19))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(13, 19))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(13, 19))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()
    plt.subplot(2, 3, 5)
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 12, nq + 18))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 12, nq + 18))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 12, nq + 18))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()
    plt.subplot(2, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 12))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(6, 12))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(6, 12))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()

    plt.figure(figIndex + 1)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[: rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(c[0])
        Cy.append(c[1])
    plt.plot(Cx, Cy)
    plt.title("CoM position")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    if show:
        plt.show()
