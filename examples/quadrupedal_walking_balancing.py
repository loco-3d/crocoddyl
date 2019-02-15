from crocoddyl import *
import numpy as np
from numpy.linalg import norm
import pinocchio



class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId

class SimpleQuadrupedalWalkingProblem:
    """ Defines a simple 3d locomotion problem
    """
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
        self.rmodel.defaultState = \
            np.concatenate([m2a(self.rmodel.neutralConfiguration),
                            np.zeros(self.rmodel.nv)])

    def createProblem(self, x0, comForward, stepLength, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple walking.

        :param x0: initial state
        :param comForward: initial CoM motion
        :param stepLength: step length
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q0)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        com0 = m2a(pinocchio.centerOfMass(self.rmodel,self.rdata,q0))
        lfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        rhFootPos0 = self.rdata.oMf[self.lhFootId].translation

        # Defining the action models along the time instances
        loco3dModel = []

        # Creating the action model for the double support phase
        doubleSupport = \
            [ self.createSwingFootModel(
                timeStep,
                [ self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId ],
                com0 + [(comForward*k)/supportKnots, 0., 0.]
                ) for k in range(supportKnots) ]
        term =  \
            self.createSwingFootModel(
                timeStep,
                [ self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId ],
                com0 + [(comForward*k)/supportKnots, 0., 0.]
                )
        term.differential.costs['comTrack'].weight = 100000

        loco3dModel += doubleSupport
        problem = ShootingProblem(x0, loco3dModel, term)
        return problem

    def createFootstepModels(self, supportFootId, swingFootId, stepLength, footPos0, numKnots):
        """ Action models for a footstep phase.

        :param supportFootId: Id of the supporting foot
        :param swingFootId: Id of the swinging foot
        :param stepLength: step length
        :param footPos0: initial position of the swinging foot
        :param numKnots: number of knots for the footstep phase
        :return footstep action models
        """
        # Action models for the foot swing
        footSwingModel = \
            [ self.createSwingFootModel(
                timeStep,
                [ supportFootId ],
                TaskSE3(
                    pinocchio.SE3(np.eye(3),
                                  np.asmatrix(a2m([ [(stepLength*k)/numKnots, 0., 0.] ]) +
                                  footPos0)),
                    swingFootId)
                ) for k in range(numKnots) ]
        # Action model for the foot switch
        footSwitchModel = \
            self.createFootSwitchModel(
                [ supportFootId ],
                TaskSE3(
                    pinocchio.SE3(np.eye(3),
                                  np.asmatrix(a2m([ stepLength, 0., 0. ]) +
                                  footPos0)),
                    swingFootId)
                )
        # Updating the current foot position for next step
        footPos0 += np.asmatrix(a2m([ stepLength, 0., 0. ]))
        return footSwingModel + [ footSwitchModel ]

    def createSwingFootModel(self, timeStep, supportFootIds, comTask = None, swingFootTask = None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating the action model for floating-base systems. A walker system 
        # is by default a floating-base system
        actModel = ActuationModelFreeFloating(self.rmodel)

        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        contactModel = ContactModelMultiple(self.rmodel)
        for i in supportFootIds:
            supportContactModel = \
                ContactModel3D(self.rmodel, i, ref=[0.,0.,0.], gains=[0.,0.])
            contactModel.addContact('contact_'+str(i), supportContactModel)

        # Creating the cost model for a contact phase
        costModel = CostModelSum(self.rmodel, actModel.nu)
        if comTask.all() != None:
            comTrack = CostModelCoM(self.rmodel, comTask, actModel.nu)
            costModel.addCost("comTrack", comTrack, 100.)
        if swingFootTask != None:
            footTrack = CostModelFramePlacement(self.rmodel,
                                                swingFootTask.frameId,
                                                swingFootTask.oXf,
                                                actModel.nu,)
            costModel.addCost("footTrack", footTrack, 100.)

        stateWeights = \
            np.array([0]*6 + [0.01]*(self.rmodel.nv-6) + [10]*self.rmodel.nv)
        stateReg = CostModelState(self.rmodel,
                                  self.state,
                                  self.rmodel.defaultState,
                                  actModel.nu,
                                  ActivationModelWeightedQuad(stateWeights**2))
        ctrlReg = CostModelControl(self.rmodel, actModel.nu)
        costModel.addCost("stateReg", stateReg, 0.1)
        costModel.addCost("ctrlReg", ctrlReg, 0.001)

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

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return action model for a foot switch phase
        """
        model = self.createSwingFootModel(0., supportFootId, swingFootTask)

        impactFootVelCost = \
            CostModelFrameVelocity(self.rmodel, swingFootTask.frameId)
        model.differential.costs.addCost('impactVel', impactFootVelCost, 10000.)
        model.differential.costs['impactVel' ].weight = 100000
        model.differential.costs['footTrack' ].weight = 100000
        model.differential.costs['stateReg'].weight = 10
        model.differential.costs['ctrlReg'].weight = 0.001
        return model



# Loading the HyQ model
hyq = loadHyQ()

# Defining the initial state of the robot
q0 = hyq.q0.copy()
v0 = pinocchio.utils.zero(hyq.model.nv)
x0 = m2a(np.concatenate([q0,v0]))

# Setting up the 3d walking problem
lfFoot = 'lf_foot'
rfFoot = 'rf_foot'
lhFoot = 'lh_foot'
rhFoot = 'rh_foot'
walk = SimpleQuadrupedalWalkingProblem(hyq.model, lfFoot, rfFoot, lhFoot, rhFoot)

# Creating the walking problem
comForward = 0.1 # meters
stepLength = 0.6 # meters
timeStep = 5e-2 # seconds
stepKnots = 20
supportKnots = 10
walkProblem = walk.createProblem(x0, comForward, stepLength, timeStep, stepKnots, supportKnots)


# Solving the 3d walking problem using DDP
ddp = SolverDDP(walkProblem)
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose(),
                CallbackSolverDisplay(hyq,4,cameraTF)]
ddp.th_stop = 1e-9
ddp.solve(maxiter=1000,regInit=.1,init_xs=[hyq.model.defaultState]*len(ddp.models()))


CallbackSolverDisplay(hyq)(ddp)

comT = pinocchio.centerOfMass(walk.rmodel,walk.rdata,a2m(ddp.callback[0].xs[-1][:walk.rmodel.nq]))
com0 = pinocchio.centerOfMass(walk.rmodel,walk.rdata,q0)
print comT - com0