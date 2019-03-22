import sys

import numpy as np
import pinocchio
from crocoddyl import (ActivationModelWeightedQuad, ActuationModelFreeFloating, CallbackDDPLogger, CallbackDDPVerbose,
                       CallbackSolverDisplay, ContactModel6D, ContactModelMultiple, CostModelCoM, CostModelControl,
                       CostModelFramePlacement, CostModelFrameVelocity, CostModelState, CostModelSum,
                       DifferentialActionModelFloatingInContact, IntegratedActionModelEuler, ShootingProblem,
                       SolverDDP, StatePinocchio, a2m, displayTrajectory, loadTalosLegs, m2a)
from pinocchio.utils import zero

WITHDISPLAY = 'disp' in sys.argv
WITHPLOT = 'plot' in sys.argv


def plotSolution(rmodel, xs, us, figIndex=1, show=True):
    import matplotlib.pyplot as plt
    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    for i in range(nx):
        X[i] = [x[i] for x in xs]
    for i in range(nu):
        U[i] = [u[i] for u in us]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    legJointNames = ['1', '2', '3', '4', '5', '6']
    # left foot
    plt.subplot(2, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 13))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 12))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 6))]
    plt.ylabel('LF')
    plt.legend()

    # right foot
    plt.subplot(2, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 19))]
    plt.ylabel('RF')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(2, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 18))]
    plt.ylabel('RF')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(2, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 12))]
    plt.ylabel('RF')
    plt.xlabel('knots')
    plt.legend()

    plt.figure(figIndex + 1)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = a2m(x[:rmodel.nq])
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
    plt.plot(Cx, Cy)
    plt.title('CoM position')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    if show:
        plt.show()


class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId


class SimpleBipedGaitProblem:
    """ Defines a simple 3d locomotion problem
    """

    def __init__(self, rmodel, rightFoot, leftFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = StatePinocchio(self.rmodel)
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = \
            np.concatenate([m2a(q0), np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Remove the armature
        self.rmodel.armature[6:] = 1.

    def createWalkingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots):
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
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        comRef = m2a(rfPos0 + lfPos0) / 2
        comRef[2] = 0.6185

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [self.createSwingFootModel(timeStep, [self.rfId, self.lfId]) for k in range(supportKnots)]

        # Creating the action models for three steps
        if self.firstStep is True:
            rStep = self.createFootstepModels(comRef, [rfPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots,
                                              [self.lfId], [self.rfId])
            self.firstStep = False
        else:
            rStep = self.createFootstepModels(comRef, [rfPos0], stepLength, stepHeight, timeStep, stepKnots,
                                              [self.lfId], [self.rfId])
        lStep = self.createFootstepModels(comRef, [lfPos0], stepLength, stepHeight, timeStep, stepKnots, [self.rfId],
                                          [self.lfId])

        # We defined the problem as:
        loco3dModel += doubleSupport + rStep
        loco3dModel += doubleSupport + lStep

        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createFootstepModels(self, comPos0, feetPos0, stepLength, stepHeight, timeStep, numKnots, supportFootIds,
                             swingFootIds):
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
                # Defining a foot swing task given the step length. The swing task
                # is decomposed on two phases: swing-up and swing-down. We decide
                # deliveratively to allocated the same number of nodes (i.e. phKnots)
                # in each phase. With this, we define a proper z-component for the
                # swing-leg motion.
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = a2m([[stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots]])
                elif k == phKnots:
                    dp = a2m([[stepLength * (k + 1) / numKnots, 0., stepHeight]])
                else:
                    dp = a2m([[stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)]])
                tref = np.asmatrix(p + dp)

                swingFootTask += [TaskSE3(pinocchio.SE3(np.eye(3), tref), i)]

            comTask = np.array([stepLength * (k + 1) / numKnots, 0., 0.]) * comPercentage + comPos0
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
            ]

        # Action model for the foot switch
        footSwitchModel = \
            self.createFootSwitchModel(supportFootIds, swingFootTask)

        # Updating the current foot position for next step
        comPos0 += np.array([stepLength * comPercentage, 0., 0.])
        for p in feetPos0:
            p += a2m([[stepLength, 0., 0.]])
        return footSwingModel + [footSwitchModel]

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
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

        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        contactModel = ContactModelMultiple(self.rmodel)
        for i in supportFootIds:
            supportContactModel = \
                ContactModel6D(self.rmodel, i, ref=pinocchio.SE3.Identity(),
                               gains=[0., 0.])
            contactModel.addContact('contact_' + str(i), supportContactModel)

        # Creating the cost model for a contact phase
        costModel = CostModelSum(self.rmodel, actModel.nu)
        if isinstance(comTask, np.ndarray):
            comTrack = CostModelCoM(self.rmodel, comTask, actModel.nu)
            costModel.addCost("comTrack", comTrack, 1e4)
        if swingFootTask is not None:
            for i in swingFootTask:
                footTrack = \
                    CostModelFramePlacement(self.rmodel,
                                            i.frameId,
                                            i.oXf,
                                            actModel.nu)
                costModel.addCost("footTrack_" + str(i), footTrack, 1e4)

        stateWeights = np.array([0] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv)
        stateReg = CostModelState(
            self.rmodel,
            self.state,
            self.rmodel.defaultState,
            actModel.nu,
            activation=ActivationModelWeightedQuad(stateWeights**2))
        ctrlReg = CostModelControl(self.rmodel, actModel.nu)
        costModel.addCost("stateReg", stateReg, 1e-1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = DifferentialActionModelFloatingInContact(self.rmodel, actModel, contactModel, costModel)
        model = IntegratedActionModelEuler(dmodel)
        model.timeStep = timeStep
        return model

    def createFootSwitchModel(self, supportFootId, swingFootTask):
        """ Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return action model for a foot switch phase
        """
        model = self.createSwingFootModel(0., supportFootId, swingFootTask=swingFootTask)

        for i in swingFootTask:
            impactFootVelCost = \
                CostModelFrameVelocity(self.rmodel, i.frameId)
            model.differential.costs.addCost('impactVel_' + str(i), impactFootVelCost, 1e4)
            model.differential.costs['impactVel_' + str(i)].weight = 1e6
            model.differential.costs['footTrack_' + str(i)].weight = 1e8
        model.differential.costs['stateReg'].weight = 1e1
        model.differential.costs['ctrlReg'].weight = 1e-3
        return model


# Creating the lower-body part of Talos
talos_legs = loadTalosLegs()
rmodel = talos_legs.model
rdata = rmodel.createData()

# Defining the initial state of the robot
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
v0 = zero(rmodel.nv)
x0 = m2a(np.concatenate([q0, v0]))

# Setting up the 3d walking problem
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'
gait = SimpleBipedGaitProblem(rmodel, rightFoot, leftFoot)

# Setting up all tasks
GAITPHASES = \
    [{'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.0375, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.0375, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.0375, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.0375, 'stepKnots': 25, 'supportKnots': 1}}]
cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'walking':
            # Creating a walking problem
            ddp[i] = SolverDDP(
                gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                          value['stepKnots'], value['supportKnots']))

    # Added the callback functions
    print('*** SOLVE ' + key + ' ***')
    ddp[i].callback = [CallbackDDPLogger(), CallbackDDPVerbose()]
    if WITHDISPLAY:
        ddp[i].callback.append(CallbackSolverDisplay(talos_legs, 4, 1, cameraTF))

    # Solving the problem with the DDP solver
    ddp[i].th_stop = 1e-9
    ddp[i].solve(
        maxiter=1000,
        regInit=.1,
        init_xs=[rmodel.defaultState] * len(ddp[i].models()),
        init_us=[
            m.differential.quasiStatic(d.differential, rmodel.defaultState)
            for m, d in zip(ddp[i].models(), ddp[i].datas())[:-1]
        ])

    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]

# Display the entire motion
if WITHDISPLAY:
    for i, phase in enumerate(GAITPHASES):
        displayTrajectory(talos_legs, ddp[i].xs, ddp[i].models()[0].timeStep)

# Plotting the entire motion
if WITHPLOT:
    xs = []
    us = []
    for i, phase in enumerate(GAITPHASES):
        xs.extend(ddp[i].xs)
        us.extend(ddp[i].us)
    plotSolution(rmodel, xs, us)
