from crocoddyl import *
import numpy as np
from numpy.linalg import norm
import pinocchio
from pinocchio.utils import *
import sys

WITHDISPLAY =  'disp' in sys.argv
WITHPLOT = 'plot' in sys.argv

class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId

class SimpleBipedWalkingProblem:
    """ Defines a simple 3d locomotion problem
    """
    def __init__(self, rmodel, rightFoot, leftFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = StatePinocchio(self.rmodel)
        self.rightFoot = rightFoot
        self.leftFoot = leftFoot
        # Defining default state
        self.rmodel.defaultState = \
            np.concatenate([m2a(self.rmodel.referenceConfigurations["half_sitting"].copy()),
                            np.zeros(self.rmodel.nv)])
        # Remove the armature
        self.rmodel.armature[6:] = 1.
    
    def createProblem(self, x0, stepLength, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple walking.

        :param x0: initial state
        :param stepLength: step length
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Getting the frame id for the right and left foot
        rightFootId = self.rmodel.getFrameId(self.rightFoot)
        leftFootId = self.rmodel.getFrameId(self.leftFoot)

        # Compute the current foot positions
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel,self.rdata,q0)
        pinocchio.updateFramePlacements(self.rmodel,self.rdata)
        rightFootPos0 = self.rdata.oMf[rightFootId].translation
        leftFootPos0 = self.rdata.oMf[leftFootId].translation

        # Defining the action models along the time instances
        loco3dModel = []
        # Creating the action models for three steps
        firstStep = self.createFootstepModels(rightFootId, leftFootId,
                                              0.5*stepLength, leftFootPos0,
                                              stepKnots)
        secondStep = self.createFootstepModels(leftFootId, rightFootId,
                                               stepLength, rightFootPos0,
                                               stepKnots)
        thirdStep = self.createFootstepModels(rightFootId, leftFootId,
                                              stepLength, leftFootPos0,
                                              stepKnots)

        # Creating the action model for the double support phase
        doubleSupport = \
            [ self.createSwingFootModel(
                timeStep,
                [ rightFootId, leftFootId ]
                ) for k in range(supportKnots) ]

        # We defined the problem as:
        #  STEP 1 - DS - STEP 2 - DS - STEP 3 - DS
        loco3dModel += firstStep + doubleSupport
        loco3dModel += secondStep + doubleSupport
        loco3dModel += thirdStep + doubleSupport

        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
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

    def createSwingFootModel(self, timeStep, supportFootIds, swingFootTask = None):
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
                ContactModel6D(self.rmodel, i, ref=pinocchio.SE3.Identity(), gains=[0.,0.])
            contactModel.addContact('contact_'+str(i), supportContactModel)

        # Creating the cost model for a contact phase
        costModel = CostModelSum(self.rmodel, actModel.nu)
        if swingFootTask != None:
            footTrack = CostModelFramePlacement(self.rmodel,
                                                swingFootTask.frameId,
                                                swingFootTask.oXf,
                                                actModel.nu)
            costModel.addCost("footTrack", footTrack, 100.)

        stateWeights = \
            np.array([0]*6 + [0.01]*(self.rmodel.nv-6) + [10]*self.rmodel.nv)
        stateReg = CostModelState(self.rmodel,
                                  self.state,
                                  self.rmodel.defaultState,
                                  actModel.nu,
                                  activation=ActivationModelWeightedQuad(stateWeights**2))
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




# Creating the lower-body part of Talos
talos_legs = loadTalosLegs()

# Defining the initial state of the robot
q = talos_legs.q0.copy()
v = zero(talos_legs.model.nv)
x0 = m2a(np.concatenate([q,v]))


# Setting up the 3d walking problem
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'
walk = SimpleBipedWalkingProblem(talos_legs.model, rightFoot, leftFoot)

# Creating the walking problem
stepLength = 0.6 # meters
timeStep = 0.0375 # seconds
stepKnots = 20
supportKnots = 10
walkProblem = walk.createProblem(x0, stepLength, timeStep, stepKnots, supportKnots)


# Solving the 3d walking problem using DDP
ddp = SolverDDP(walkProblem)
cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
ddp.callback = [ CallbackDDPVerbose() ]
if WITHPLOT:       ddp.callback.append(CallbackDDPLogger())
if WITHDISPLAY:    ddp.callback.append(CallbackSolverDisplay(talos_legs,4,1,cameraTF))
ddp.th_stop = 1e-9
ddp.solve(maxiter=1000,regInit=.1,init_xs=[talos_legs.model.defaultState]*len(ddp.models()))


# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = ddp.callback[0]
    plotOCSolution(log.xs, log.us)
    plotDDPConvergence(log.costs,log.control_regs,
                       log.state_regs,log.gm_stops,
                       log.th_stops,log.steps)

# Visualization of the DDP solution in gepetto-viewer
if WITHDISPLAY:
    CallbackSolverDisplay(talos_legs)(ddp)