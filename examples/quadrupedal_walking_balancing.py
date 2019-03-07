import sys

import numpy as np
from numpy.linalg import norm

import pinocchio
from crocoddyl import *

WITHDISPLAY =  'disp' in sys.argv
WITHPLOT = 'plot' in sys.argv

def plotSolution(xs, us):
    import matplotlib.pyplot as plt
    # Getting the state and control trajectories
    nx = xs[0].shape[0]
    nu = us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    for i in range(nx):
        X[i] = [x[i] for x in xs]
    for i in range(nu):
        U[i] = [u[i] for u in us]

    plt.figure(1)

    # Plotting the joint trajectories and torque
    legJointNames = ['HAA', 'HFE', 'KFE']
    # LF foot
    plt.subplot(421)
    [plt.plot(X[k], label=legJointNames[i]) for i,k in enumerate(range(7,10))]
    plt.title('LF')
    plt.legend()
    plt.subplot(422)
    [plt.plot(U[k], label=legJointNames[i]) for i,k in enumerate(range(0,3))]
    plt.title('LF')
    plt.legend()

    # LH foot
    plt.subplot(423)
    [plt.plot(X[k], label=legJointNames[i]) for i,k in enumerate(range(10,13))]
    plt.title('LH')
    plt.legend()
    plt.subplot(424)
    [plt.plot(U[k], label=legJointNames[i]) for i,k in enumerate(range(3,6))]
    plt.title('LH')
    plt.legend()

    # RF foot
    plt.subplot(425)
    [plt.plot(X[k], label=legJointNames[i]) for i,k in enumerate(range(13,16))]
    plt.title('RF')
    plt.legend()
    plt.subplot(426)
    [plt.plot(U[k], label=legJointNames[i]) for i,k in enumerate(range(6,9))]
    plt.title('RF')
    plt.legend()

    # RH foot
    plt.subplot(427)
    [plt.plot(X[k], label=legJointNames[i]) for i,k in enumerate(range(16,19))]
    plt.title('RH')
    plt.legend()
    plt.subplot(428)
    [plt.plot(U[k], label=legJointNames[i]) for i,k in enumerate(range(9,12))]
    plt.title('RH')
    plt.legend()
    plt.xlabel('knots')
    plt.show()


class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId


class SimpleQuadrupedalGaitProblem:
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = StatePinocchio(self.rmodel)
        # Getting the frame id for all the legs
        self.lfFootId = self.rmodel.getFrameId(lfFoot)
        self.rfFootId = self.rmodel.getFrameId(rfFoot)
        self.lhFootId = self.rmodel.getFrameId(lhFoot)
        self.rhFootId = self.rmodel.getFrameId(rhFoot)
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = \
            np.concatenate([m2a(q0), np.zeros(self.rmodel.nv)])
        self.firstStep = True

    def createCoMProblem(self, x0, comGoTo, timeStep, numKnots):
        """ Create a shooting problem for a CoM forward/backward task.

        :param x0: initial state
        :param comGoTo: initial CoM motion
        :param timeStep: step time for each knot
        :param numKnots: number of knots per each phase
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation

        # Defining the action models along the time instances
        comModels = []

        # Creating the action model for the CoM task
        comForwardModels = \
            [self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(numKnots)]
        comForwardTermModel =  \
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                com0 + [comGoTo, 0., 0.]
            )
        comForwardTermModel.differential.costs['comTrack'].weight = 1e6

        comBackwardModels = \
            [self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(numKnots)]
        comBackwardTermModel =  \
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                com0 + [-comGoTo, 0., 0.]
            )
        comBackwardTermModel.differential.costs['comTrack'].weight = 1e6

        # Adding the CoM tasks
        comModels += comForwardModels + [comForwardTermModel]
        comModels += comBackwardModels + [comBackwardTermModel]

        # Defining the shooting problem
        problem = ShootingProblem(x0, comModels, comModels[-1])
        return problem

    def createWalkingProblem(self, x0, stepLength, stepHeight,
                             timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        comPos0 = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = \
            [self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(supportKnots)]
        if self.firstStep is True:
            rhStep = \
                self.createFootstepModels(
                    comPos0, [rhFootPos0],
                    0.5*stepLength, stepHeight, timeStep, stepKnots,
                    [self.lfFootId, self.rfFootId, self.lhFootId],
                    [self.rhFootId])
            rfStep = \
                self.createFootstepModels(
                    comPos0, [rfFootPos0],
                    0.5*stepLength, stepHeight, timeStep, stepKnots,
                    [self.lfFootId, self.lhFootId, self.rhFootId],
                    [self.rfFootId])
            self.firstStep = False
        else:
            rhStep = \
                self.createFootstepModels(
                    comPos0, [rhFootPos0],
                    stepLength, stepHeight, timeStep, stepKnots,
                    [self.lfFootId, self.rfFootId, self.lhFootId],
                    [self.rhFootId])
            rfStep = \
                self.createFootstepModels(
                    comPos0, [rfFootPos0],
                    stepLength, stepHeight, timeStep, stepKnots,
                    [self.lfFootId, self.lhFootId, self.rhFootId],
                    [self.rfFootId])
        lhStep = \
            self.createFootstepModels(
                comPos0, [lhFootPos0],
                stepLength, stepHeight, timeStep, stepKnots,
                [self.lfFootId, self.rfFootId, self.rhFootId],
                [self.lhFootId])
        lfStep = \
            self.createFootstepModels(
                comPos0, [lfFootPos0],
                stepLength, stepHeight, timeStep, stepKnots,
                [self.rfFootId, self.lhFootId, self.rhFootId],
                [self.lfFootId])

        loco3dModel += doubleSupport + rhStep + rfStep
        loco3dModel += doubleSupport + lhStep + lfStep

        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createTrottingProblem(self, x0, stepLength, stepHeight,
                              timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple trotting gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        comPos0 = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = \
            [self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(supportKnots)]
        if self.firstStep is True:
            rflhStep = \
                self.createFootstepModels(
                    comPos0, [rfFootPos0, lhFootPos0],
                    0.5*stepLength, stepHeight, timeStep, stepKnots,
                    [self.lfFootId, self.rhFootId],
                    [self.rfFootId, self.lhFootId])
            self.firstStep = False
        else:
            rflhStep = \
                self.createFootstepModels(
                    comPos0, [rfFootPos0, lhFootPos0],
                    stepLength, stepHeight, timeStep, stepKnots,
                    [self.lfFootId, self.rhFootId],
                    [self.rfFootId, self.lhFootId])
        lfrhStep = \
            self.createFootstepModels(
                comPos0, [lfFootPos0, rhFootPos0],
                stepLength, stepHeight, timeStep, stepKnots,
                [self.rfFootId, self.lhFootId],
                [self.lfFootId, self.rhFootId])

        loco3dModel += doubleSupport + rflhStep
        loco3dModel += doubleSupport + lfrhStep

        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createPacingProblem(self, x0, stepLength, stepHeight,
                            timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple pacing gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        comPos0 = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = \
            [self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(supportKnots)]
        if self.firstStep is True:
            rightSteps = \
                self.createFootstepModels(
                    comPos0, [rfFootPos0, rhFootPos0],
                    0.5*stepLength, stepHeight, timeStep, stepKnots,
                    [self.lfFootId, self.lhFootId],
                    [self.rfFootId, self.rhFootId])
            self.firstStep = False
        else:
            rightSteps = \
                self.createFootstepModels(
                    comPos0, [rfFootPos0, rhFootPos0],
                    stepLength, stepHeight, timeStep, stepKnots,
                    [self.lfFootId, self.lhFootId],
                    [self.rfFootId, self.rhFootId])
        leftSteps = \
            self.createFootstepModels(
                comPos0, [lfFootPos0, lhFootPos0],
                stepLength, stepHeight, timeStep, stepKnots,
                [self.rfFootId, self.rhFootId],
                [self.lfFootId, self.lhFootId])

        loco3dModel += doubleSupport + rightSteps
        loco3dModel += doubleSupport + leftSteps

        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createBoundingProblem(self, x0, stepLength, stepHeight,
                              timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple bounding gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        comPos0 = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = \
            [self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId])
                for k in range(supportKnots)]
        hindSteps = \
            self.createFootstepModels(
                comPos0, [lfFootPos0, rfFootPos0],
                stepLength, stepHeight, timeStep, stepKnots,
                [self.lhFootId, self.rhFootId],
                [self.lfFootId, self.rfFootId])
        frontSteps = \
            self.createFootstepModels(
                comPos0, [lhFootPos0, rhFootPos0],
                stepLength, stepHeight, timeStep, stepKnots,
                [self.lfFootId, self.rfFootId],
                [self.lhFootId, self.rhFootId])

        loco3dModel += doubleSupport + hindSteps
        loco3dModel += doubleSupport + frontSteps

        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createJumpingProblem(self, x0, jumpHeight, timeStep):
        comPos0 = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation

        takeOffKnots = 30
        flyingKnots = 30

        loco3dModel = []
        takeOff = \
            [self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(takeOffKnots)]
        flyingPhase = \
            [self.createSwingFootModel(
                timeStep,
                [],
                np.array([0., 0., jumpHeight * (k+1) / flyingKnots]) + comPos0
            ) for k in range(flyingKnots)]

        loco3dModel += takeOff
        loco3dModel += flyingPhase

        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createFootstepModels(self, comPos0, feetPos0, stepLength, stepHeight,
                             timeStep, numKnots, supportFootIds, swingFootIds):
        """ Action models for a footstep phase.

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
                resKnot = numKnots % 2
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = a2m([[stepLength * (k+1) / numKnots, 0.,
                               stepHeight * k / phKnots]])
                elif k == phKnots:
                    dp = a2m([[stepLength * (k+1) / numKnots, 0., stepHeight]])
                else:
                    dp = a2m([[stepLength * (k+1) / numKnots, 0.,
                             stepHeight * (1 - float(k-phKnots) / phKnots)]])
                tref = np.asmatrix(p + dp)

                swingFootTask += \
                    [TaskSE3(pinocchio.SE3(np.eye(3), tref), i)]

            # Adding an action model for this knot
            comTask = \
                np.array([stepLength * (k+1) / numKnots, 0., 0.]) * \
                comPercentage + comPos0
            footSwingModel += \
                [self.createSwingFootModel(timeStep, supportFootIds,
                                           comTask=comTask,
                                           swingFootTask=swingFootTask)]
        # Action model for the foot switch
        footSwitchModel = \
            self.createFootSwitchModel(supportFootIds, swingFootTask)

        # Updating the current foot position for next step
        comPos0 += np.array([stepLength * comPercentage, 0., 0.])
        for p in feetPos0:
            p += a2m([[stepLength, 0., 0.]])
        return footSwingModel + [footSwitchModel]

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None,
                             swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating the action model for floating-base systems. A walker system
        # is by default a floating-base system
        actModel = ActuationModelFreeFloating(self.rmodel)

        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        contactModel = ContactModelMultiple(self.rmodel)
        for i in supportFootIds:
            supportContactModel = \
                ContactModel3D(self.rmodel, i, ref=[
                               0., 0., 0.], gains=[0., 0.])
            contactModel.addContact('contact_'+str(i), supportContactModel)

        # Creating the cost model for a contact phase
        costModel = CostModelSum(self.rmodel, actModel.nu)
        if isinstance(comTask, np.ndarray):
            comTrack = CostModelCoM(self.rmodel, comTask, actModel.nu)
            costModel.addCost("comTrack", comTrack, 1e4)
        if swingFootTask is not None:
            for i in swingFootTask:
                footTrack = \
                    CostModelFrameTranslation(self.rmodel,
                                              i.frameId,
                                              m2a(i.oXf.translation),
                                              actModel.nu)
                costModel.addCost("footTrack_"+str(i), footTrack, 1e4)

        stateWeights = \
            np.array([0]*3 + [500.]*3 + [0.01]*(self.rmodel.nv-6) +
                     [10]*self.rmodel.nv)
        stateReg = CostModelState(self.rmodel,
                                  self.state,
                                  self.rmodel.defaultState,
                                  actModel.nu,
                                  ActivationModelWeightedQuad(stateWeights**2))
        ctrlReg = CostModelControl(self.rmodel, actModel.nu)
        costModel.addCost("stateReg", stateReg, 1e-1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-4)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = \
            DifferentialActionModelFloatingInContact(self.rmodel,
                                                     actModel,
                                                     contactModel,
                                                     costModel)
        model = IntegratedActionModelEuler(dmodel)
        model.timeStep = timeStep
        return model

    def createFootSwitchModel(self, supportFootId, swingFootTask):
        """ Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return action model for a foot switch phase
        """
        model = self.createSwingFootModel(
            0., supportFootId, swingFootTask=swingFootTask)

        for i in swingFootTask:
            impactFootVelCost = \
                CostModelFrameVelocity(self.rmodel, i.frameId)
            model.differential.costs.addCost('impactVel_'+str(i),
                                             impactFootVelCost, 1e4)
            model.differential.costs['impactVel_'+str(i)].weight = 1e6
            model.differential.costs['footTrack_'+str(i)].weight = 1e6
        model.differential.costs['stateReg'].weight = 1e1
        model.differential.costs['ctrlReg'].weight = 1e-3
        return model


# Loading the HyQ model
hyq = loadHyQ()
if WITHDISPLAY: hyq.initDisplay(loadModel=True)

rmodel = hyq.model
rdata  = rmodel.createData()

# Defining the initial state of the robot
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
v0 = pinocchio.utils.zero(rmodel.nv)
x0 = m2a(np.concatenate([q0, v0]))

# Setting up the 3d walking problem
lfFoot = 'lf_foot'
rfFoot = 'rf_foot'
lhFoot = 'lh_foot'
rhFoot = 'rh_foot'
gait = SimpleQuadrupedalGaitProblem(
    rmodel, lfFoot, rfFoot, lhFoot, rhFoot)

# Setting up all tasks
GAITPHASES = \
    [{'walking': {'stepLength': 0.15, 'stepHeight': 0.2,
                  'timeStep': 1e-2, 'stepKnots': 25, 'supportKnots': 5}},
     {'trotting': {'stepLength': 0.15, 'stepHeight': 0.2,
                   'timeStep': 1e-2, 'stepKnots': 25, 'supportKnots': 5}},
     {'pacing': {'stepLength': 0.15, 'stepHeight': 0.2,
                 'timeStep': 1e-2, 'stepKnots': 25, 'supportKnots': 5}},
     {'bounding': {'stepLength': 0.15, 'stepHeight': 0.2,
                   'timeStep': 1e-2, 'stepKnots': 25, 'supportKnots': 5}},
     {'jumping': {'jumpHeight': 0.5, 'timeStep': 1e-2}}]
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key is 'walking':
            # Creating a walking problem
            ddp[i] = SolverDDP(
                gait.createWalkingProblem(
                    x0, value['stepLength'], value['stepHeight'],
                    value['timeStep'],
                    value['stepKnots'], value['supportKnots']))
        elif key is 'trotting':
            # Creating a trotting problem
            ddp[i] = SolverDDP(
                gait.createTrottingProblem(
                    x0, value['stepLength'], value['stepHeight'],
                    value['timeStep'],
                    value['stepKnots'], value['supportKnots']))
        elif key is 'pacing':
            # Creating a pacing problem
            ddp[i] = SolverDDP(
                gait.createPacingProblem(
                    x0, value['stepLength'], value['stepHeight'],
                    value['timeStep'],
                    value['stepKnots'], value['supportKnots']))
        elif key is 'bounding':
            # Creating a bounding problem
            ddp[i] = SolverDDP(
                gait.createBoundingProblem(
                    x0, value['stepLength'], value['stepHeight'],
                    value['timeStep'],
                    value['stepKnots'], value['supportKnots']))
        elif key is 'jumping':
            # Creating a jumping problem
            ddp[i] = SolverDDP(
                gait.createJumpingProblem(
                    x0, value['jumpHeight'], value['timeStep']))

    # Added the callback functions
    print
    print 'Solving ' + key
    ddp[i].callback = [CallbackDDPLogger(), CallbackDDPVerbose()]
    if WITHDISPLAY:
        ddp[i].callback.append(CallbackSolverDisplay(hyq, 4, 1, cameraTF))

    # Solving the problem with the DDP solver
    ddp[i].th_stop = 1e-9
    ddp[i].solve(
        maxiter=1000, regInit=.1,
        init_xs=[rmodel.defaultState]*len(ddp[i].models()),
        init_us=[m.differential.quasiStatic(d.differential,
                                            rmodel.defaultState)
                 for m, d in zip(ddp[i].models(), ddp[i].datas())[:-1]])

    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]


# Display the entire motion
if WITHDISPLAY:
    for i, phase in enumerate(GAITPHASES):
        displayTrajectory(hyq, ddp[i].xs, ddp[i].models()[0].timeStep)


# Plotting the entire motion
if WITHPLOT:
    for i, phase in enumerate(GAITPHASES):
        log = ddp[i].callback[0]
        plotSolution(log.xs, log.us)
        plotDDPConvergence(log.costs,log.control_regs,
                           log.state_regs,log.gm_stops,
                           log.th_stops,log.steps)
