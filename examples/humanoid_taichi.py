import os
import sys

import crocoddyl
from crocoddyl.utils.biped import plotSolution
import numpy as np
import example_robot_data
import pinocchio

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Load robot
robot = example_robot_data.load('talos')
rmodel = robot.model
lims = rmodel.effortLimit
# lims[19:] *= 0.5  # reduced artificially the torque limits
rmodel.effortLimit = lims

# Create data structures
rdata = rmodel.createData()
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Set integration time
DT = 5e-2
T = 40
target = np.array([0.4, 0, 1.2])

# Initialize reference state, target and reference CoM
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'
endEffector = 'gripper_left_joint'
endEffectorId = rmodel.getFrameId(endEffector)
rightFootId = rmodel.getFrameId(rightFoot)
leftFootId = rmodel.getFrameId(leftFoot)
q0 = rmodel.referenceConfigurations["half_sitting"]
rmodel.defaultState = np.concatenate([q0, np.zeros(rmodel.nv)])
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
rfPos0 = rdata.oMf[rightFootId].translation
lfPos0 = rdata.oMf[leftFootId].translation
refGripper = rdata.oMf[rmodel.getFrameId("gripper_left_joint")].translation
comRef = (rfPos0 + lfPos0) / 2
comRef[2] = np.asscalar(pinocchio.centerOfMass(rmodel, rdata, q0)[2])

# Initialize Gepetto viewer
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, frameNames=[rightFoot, leftFoot])
    display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
    display.robot.viewer.gui.applyConfiguration('world/point', target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion

# Create two contact models used along the motion
contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
framePlacementLeft = crocoddyl.FramePlacement(leftFootId, pinocchio.SE3.Identity())
framePlacementRight = crocoddyl.FramePlacement(rightFootId, pinocchio.SE3.Identity())
supportContactModelLeft = crocoddyl.ContactModel6D(state, framePlacementLeft, actuation.nu, np.array([0, 40]))
supportContactModelRight = crocoddyl.ContactModel6D(state, framePlacementRight, actuation.nu, np.array([0, 40]))
contactModel1Foot.addContact(rightFoot + "_contact", supportContactModelRight)
contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)

# Cost for self-collision
maxfloat = sys.float_info.max
xlb = np.concatenate([
    -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
    rmodel.lowerPositionLimit[7:],
    -maxfloat * np.ones(state.nv)
])
xub = np.concatenate([
    maxfloat * np.ones(6),  # dimension of the SE(3) manifold
    rmodel.upperPositionLimit[7:],
    maxfloat * np.ones(state.nv)
])
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.)
limitCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuadraticBarrier(bounds), rmodel.defaultState,
                                     actuation.nu)

# Cost for state and control
stateWeights = np.array([0] * 3 + [10.] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)
stateWeightsTerm = np.array([0] * 3 + [10.] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv)
xRegCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2), rmodel.defaultState,
                                    actuation.nu)
uRegCost = crocoddyl.CostModelControl(state, actuation.nu)
xRegTermCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeightsTerm**2),
                                        rmodel.defaultState, actuation.nu)

# Cost for target reaching: hand and foot
handTrackingWeights = np.array([1] * 3 + [0.0001] * 3)
Pref = crocoddyl.FramePlacement(endEffectorId, pinocchio.SE3(np.eye(3), target))
handTrackingCost = crocoddyl.CostModelFramePlacement(state,
                                                     crocoddyl.ActivationModelWeightedQuad(handTrackingWeights**2),
                                                     Pref, actuation.nu)

footTrackingWeights = np.array([1, 1, 0.1] + [1.] * 3)
Pref = crocoddyl.FramePlacement(leftFootId, pinocchio.SE3(np.eye(3), np.array([0., 0.4, 0.])))
footTrackingCost1 = crocoddyl.CostModelFramePlacement(state,
                                                      crocoddyl.ActivationModelWeightedQuad(footTrackingWeights**2),
                                                      Pref, actuation.nu)
Pref = crocoddyl.FramePlacement(leftFootId, pinocchio.SE3(np.eye(3), np.array([0.3, 0.15, 0.35])))
footTrackingCost2 = crocoddyl.CostModelFramePlacement(state,
                                                      crocoddyl.ActivationModelWeightedQuad(footTrackingWeights**2),
                                                      Pref, actuation.nu)

# Cost for CoM reference
comTrack = crocoddyl.CostModelCoMPosition(state, comRef, actuation.nu)

# Create cost model per each action model. We divide the motion in 3 phases plus its terminal model
runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

# Then let's added the running and terminal cost functions
runningCostModel1.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel1.addCost("limitCost", limitCost, 1e3)

runningCostModel2.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel2.addCost("footPose", footTrackingCost1, 1e1)
runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel2.addCost("limitCost", limitCost, 1e3)

runningCostModel3.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel3.addCost("footPose", footTrackingCost2, 1e1)
runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel3.addCost("limitCost", limitCost, 1e3)

terminalCostModel.addCost("gripperPose", handTrackingCost, 1e2)
terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
terminalCostModel.addCost("limitCost", limitCost, 1e3)

# Create the action model
dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel2Feet,
                                                                     runningCostModel1)
dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                     runningCostModel2)
dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                     runningCostModel3)
dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel1Foot,
                                                                     terminalCostModel)

runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

# Problem definition
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverBoxFDDP(problem)
if WITHDISPLAY and WITHPLOT:
    ddp.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackDisplay(crocoddyl.GepettoDisplay(robot, 4, 4, frameNames=[rightFoot, leftFoot]))
    ])
elif WITHDISPLAY:
    ddp.setCallbacks([
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackDisplay(crocoddyl.GepettoDisplay(robot, 4, 4, frameNames=[rightFoot, leftFoot]))
    ])
elif WITHPLOT:
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    ddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
xs = [rmodel.defaultState] * (ddp.problem.T + 1)
us = ddp.problem.quasiStatic([rmodel.defaultState] * ddp.problem.T)
ddp.th_stop = 1e-7
ddp.solve(xs, us, 500, False, 1e-9)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    display.displayFromSolver(ddp)

# Get final state and end effector position
xT = ddp.xs[-1]
pinocchio.forwardKinematics(rmodel, rdata, xT[:state.nq])
pinocchio.updateFramePlacements(rmodel, rdata)
com = pinocchio.centerOfMass(rmodel, rdata, xT[:state.nq])
finalPosEff = np.array(rdata.oMf[rmodel.getFrameId("gripper_left_joint")].translation.T.flat)

print('Finally reached = ', finalPosEff)
print('Distance between hand and target = ', np.linalg.norm(finalPosEff - target))
print('Distance to default state = ', np.linalg.norm(rmodel.defaultState - np.array(xT.flat)))
print('XY distance to CoM reference = ', np.linalg.norm(com[:2] - comRef[:2]))

# Plotting the entire motion
if WITHPLOT:
    log = ddp.getCallbacks()[0]
    plotSolution(ddp, bounds=False, figIndex=1, show=False)

    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=3)
