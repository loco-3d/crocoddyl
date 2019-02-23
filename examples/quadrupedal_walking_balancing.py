from crocoddyl import *
import numpy as np
from numpy.linalg import norm
import pinocchio
import sys

WITHDISPLAY =  'disp' in sys.argv

class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId


class SimpleQuadrupedalWalkingProblem:
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

    def createWalkingProblem(self, x0, stepLength, timeStep, stepKnots,
                             supportKnots):
        """ Create a shooting problem for a simple walking.

        :param x0: initial state
        :param stepLength: step length
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))
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
        rhStep = \
            self.createFootstepModels(
                [self.lfFootId, self.rfFootId, self.lhFootId],
                [self.rhFootId],
                0.5*stepLength, [rhFootPos0], stepKnots)
        rfStep = \
            self.createFootstepModels(
                [self.lfFootId, self.lhFootId, self.rhFootId],
                [self.rfFootId],
                0.5*stepLength, [rfFootPos0], stepKnots)
        lhStep = \
            self.createFootstepModels(
                [self.lfFootId, self.rfFootId, self.rhFootId],
                [self.lhFootId],
                stepLength, [lhFootPos0], stepKnots)
        lfStep = \
            self.createFootstepModels(
                [self.rfFootId, self.lhFootId, self.rhFootId],
                [self.lfFootId],
                stepLength, [lfFootPos0], stepKnots)

        loco3dModel += doubleSupport + rhStep + rfStep
        loco3dModel += doubleSupport + lhStep + lfStep

        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createFootstepModels(self, supportFootId, swingFootIds, stepLength,
                             footPos0, numKnots):
        """ Action models for a footstep phase.

        :param supportFootId: Ids of the supporting feet
        :param swingFootId: Ids of the swinging foot
        :param stepLength: step length
        :param footPos0: initial position of the swinging foot
        :param numKnots: number of knots for the footstep phase
        :return footstep action models
        """
        # Action models for the foot swing
        footSwingModel = []
        swingFootTask = []
        for k in range(numKnots):
            for i, p in zip(swingFootIds,footPos0):
                # Defining a foot swing task given the step length
                tref = np.asmatrix(
                    a2m([[(stepLength*(k+1))/numKnots, 0., 0.]]) + p)
                swingFootTask += \
                    [TaskSE3(pinocchio.SE3(np.eye(3), tref), i)]

            # Adding an action model for this knot
            footSwingModel += \
                [self.createSwingFootModel(timeStep, supportFootId,
                                           swingFootTask=swingFootTask)]
        # Action model for the foot switch
        footSwitchModel = \
            self.createFootSwitchModel(supportFootId, swingFootTask)

        # Updating the current foot position for next step
        footPos0 = tref
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
            costModel.addCost("comTrack", comTrack, 1e2)
        if swingFootTask is not None:
            for i in swingFootTask:
                footTrack = \
                    CostModelFrameTranslation(self.rmodel,
                                              i.frameId,
                                              m2a(i.oXf.translation),
                                              actModel.nu)
                costModel.addCost("footTrack_"+str(i), footTrack, 1e2)

        stateWeights = \
            np.array([0]*6 + [0.01]*(self.rmodel.nv-6) + [10]*self.rmodel.nv)
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
            model.differential.costs.addCost('impactVel_'+str(i), impactFootVelCost, 1e4)
            model.differential.costs['impactVel_'+str(i)].weight = 1e5
            model.differential.costs['footTrack_'+str(i)].weight = 1e5
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
walk = SimpleQuadrupedalWalkingProblem(
    rmodel, lfFoot, rfFoot, lhFoot, rhFoot)

# Setting up the walking variables
stepLength = 0.35  # meters
timeStep = 5e-2  # seconds
stepKnots = 5
supportKnots = 2
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]


# Creating the walking problem and solver
ddp1 = SolverDDP(
    walk.createWalkingProblem(
        x0, stepLength, timeStep, stepKnots, supportKnots))
ddp1.callback = [CallbackDDPLogger(), CallbackDDPVerbose()]
if WITHDISPLAY:    ddp1.callback.append(CallbackSolverDisplay(hyq, 4, 1, cameraTF))
ddp1.th_stop = 1e-9
ddp1.solve(maxiter=1000, regInit=.1,
           init_xs=[rmodel.defaultState]*len(ddp1.models()),
           init_us=[m.differential.quasiStatic(d.differential, rmodel.defaultState)
                    for m, d in zip(ddp1.models(), ddp1.datas())[:-1]])

# Creating the CoM forward/backward task
comGoTo = 0.1  # meters
x0 = ddp1.xs[-1]
supportKnots = 5
ddp2 = SolverDDP(
    walk.createCoMProblem(
        x0, comGoTo, timeStep, supportKnots))
ddp2.callback = [CallbackDDPLogger(), CallbackDDPVerbose()]
if WITHDISPLAY:    ddp2.callback.append(CallbackSolverDisplay(hyq, 4, 1, cameraTF))
ddp2.th_stop = 1e-9
ddp2.solve(maxiter=1000, regInit=.1,
           init_xs=[x0]*len(ddp2.models()),
           init_us=[m.differential.quasiStatic(d.differential, x0)
                    for m, d in zip(ddp2.models(), ddp2.datas())[:-1]])


# Display the entire motion
if WITHDISPLAY:
    from crocoddyl.diagnostic import displayTrajectory
    displayTrajectory(hyq,ddp1.xs,ddp1.models()[0].timeStep)
    displayTrajectory(hyq,ddp1.xs,ddp2.models()[0].timeStep)
