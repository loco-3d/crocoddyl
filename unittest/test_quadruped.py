import numpy as np
from numpy.linalg import norm

import pinocchio
from crocoddyl import (ActivationModelWeightedQuad, ActuationModelFreeFloating, CallbackDDPVerbose, ContactModel3D,
                       ContactModelMultiple, CostModelCoM, CostModelControl, CostModelFrameTranslation, CostModelState,
                       CostModelSum, DifferentialActionModelFloatingInContact, IntegratedActionModelEuler,
                       ShootingProblem, SolverDDP, StatePinocchio, a2m, loadHyQ, m2a)

## This is an integrative test where we checked that the DDP solver generates
## a CoM motion for the HyQ robot as requested.


class SimpleQuadrupedProblem:
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
        self.rmodel.defaultState = np.concatenate(
            [m2a(self.rmodel.referenceConfigurations["half_sitting"]),
             np.zeros(self.rmodel.nv)])

    def createProblem(self, x0, comGoTo, timeStep, numKnots):
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
        comForwardModels = [
            self.createModels(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(numKnots)
        ]
        comForwardTermModel = self.createModels(timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                                com0 + [comGoTo, 0., 0.])
        comForwardTermModel.differential.costs['comTrack'].weight = 1e6

        # Adding the CoM tasks
        comModels += comForwardModels + [comForwardTermModel]

        # Defining the shooting problem
        problem = ShootingProblem(x0, comModels, comModels[-1])
        return problem

    def createModels(self, timeStep, supportFootIds, comTask=None):
        # Creating the action model for floating-base systems
        actModel = ActuationModelFreeFloating(self.rmodel)

        # Creating a 3D multi-contact model, and then including the supporting
        # feet
        contactModel = ContactModelMultiple(self.rmodel)
        for i in supportFootIds:
            supportContactModel = ContactModel3D(self.rmodel, i, ref=[0., 0., 0.], gains=[0., 0.])
            contactModel.addContact('contact_' + str(i), supportContactModel)

        # Creating the cost model for a contact phase
        costModel = CostModelSum(self.rmodel, actModel.nu)

        # CoM tracking cost
        if isinstance(comTask, np.ndarray):
            comTrack = CostModelCoM(self.rmodel, comTask, actModel.nu)
            costModel.addCost("comTrack", comTrack, 1e2)

        # State and control regularization
        stateWeights = np.array([0] * 6 + [0.01] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv)
        stateReg = CostModelState(self.rmodel, self.state, self.rmodel.defaultState, actModel.nu,
                                  ActivationModelWeightedQuad(stateWeights**2))
        ctrlReg = CostModelControl(self.rmodel, actModel.nu)
        costModel.addCost("stateReg", stateReg, 1e-1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-4)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = DifferentialActionModelFloatingInContact(self.rmodel, actModel, contactModel, costModel)
        model = IntegratedActionModelEuler(dmodel)
        model.timeStep = timeStep
        return model


# Loading the HyQ model
robot = loadHyQ()
rmodel = robot.model
rdata = rmodel.createData()

# Defining the initial state of the robot
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
v0 = pinocchio.utils.zero(rmodel.nv)
x0 = m2a(np.concatenate([q0, v0]))

# Setting up the 3d walking problem
lfFoot = 'lf_foot'
rfFoot = 'rf_foot'
lhFoot = 'lh_foot'
rhFoot = 'rh_foot'
walk = SimpleQuadrupedProblem(rmodel, lfFoot, rfFoot, lhFoot, rhFoot)

# Setting up the walking variables
comGoTo = 0.1  # meters
timeStep = 5e-2  # seconds
supportKnots = 2

# Creating the CoM problem and solving it
ddp = SolverDDP(walk.createProblem(x0, comGoTo, timeStep, supportKnots))
#ddp.callback = [ CallbackDDPVerbose() ]
ddp.th_stop = 1e-9
ddp.solve(maxiter=1000, regInit=.1, init_xs=[rmodel.defaultState] * len(ddp.models()))

# Checking the CoM displacement
x0 = ddp.xs[0]
xT = ddp.xs[-1]
q0 = a2m(x0[:rmodel.nq])
qT = a2m(xT[:rmodel.nq])
data0 = rmodel.createData()
dataT = rmodel.createData()
comT = pinocchio.centerOfMass(rmodel, dataT, qT)
com0 = pinocchio.centerOfMass(rmodel, data0, q0)
assert (norm(comT - com0 - np.matrix([[comGoTo], [0.], [0.]])) < 1e-3)
