"""
The goal of this tutorial is to demonstrate loading a robot from an urdf and controlling it.
We will use a double pendulum robot (full actuated) and control it like an acrobot (under actuated).

Example usage:
> python3 simple_urdf.py
> python3 simple_urdf.py plot
> python3 simple_urdf.py plot display
"""

import os
import sys
import numpy as np
import pathlib
import crocoddyl
import pinocchio

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

dt = 1e-3  # Time step
T = 1000  # Number of knots

# Path to the urdf

urdf_model_path = pathlib.Path('robots', 'double_pendulum_simple.urdf')
urdf_model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), urdf_model_path)
robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(str(urdf_model_path))

# The model loaded from urdf (via pinicchio)
# Note it has nu = 2 but we want an acrobot with nu=1 so we will have to define an actuation mapping later
print(robot.model)

# Create a multibody state from the pinocchio model.
state = crocoddyl.StateMultibody(robot.model)


# Define the control signal to actuated joint mapping
class AcrobotActuationModel(crocoddyl.ActuationModelAbstract):

    def __init__(self, state):
        nu = 1  # Control dimension
        crocoddyl.ActuationModelAbstract.__init__(self, state, nu=nu)

    def calc(self, data, x, u):
        # Map the control dimensions to the joint torque
        data.tau[0] = 0
        data.tau[1] = u

    def calcDiff(self, data, x, u):
        data.dtau_du[0] = 0
        data.dtau_du[1] = 1


# Also see ActuationModelFloatingBase and ActuationModelFull
actuationModel = AcrobotActuationModel(state)
# actuationModel = crocoddyl.ActuationModelNumDiff(actuationModel) Doesnt appear to be exposed in python

# Cost models
runningCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)

# Add a cost for the end effector frame position
# Note: This could be done more simply by using the joint angles
# Using the end effector frame from the urdf
eeid = robot.model.getFrameId('link3')
pref = pinocchio.SE3.Identity()  # referece frame placement
pref.translation[2] = 0.3
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, id=eeid, pref=pref, nu=actuationModel.nu)
frameCostModel = crocoddyl.CostModelResidual(state, framePlacementResidual)
runningCostModel.addCost("ee_frame_cost", cost=frameCostModel, weight=1e-7 / dt)
terminalCostModel.addCost("ee_frame_cost", cost=frameCostModel, weight=1000)

# Add a cost on control
controlResidual = crocoddyl.ResidualModelControl(state, nu=actuationModel.nu)
bounds = crocoddyl.ActivationBounds(np.array([-1.]), np.array([1.]))
activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
controlCost = crocoddyl.CostModelResidual(state, activation=activation, residual=controlResidual)
runningCostModel.addCost("control_cost", cost=controlCost, weight=1e-2 / dt)

# Create the action models for the state
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModel), 0.)

# Define a shooting problem
q0 = np.zeros((state.nq, ))  # Inital joint configurations
q0[0] = np.pi / 2  # Down
v0 = np.zeros((state.nv, ))  # Initial joint velocities
x0 = np.concatenate((q0, v0))  # Inital robot state
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Test with a rollout
if True:
    # Test the problem with a rollout
    us = [0.01 * np.ones((1, ))] * T
    xs = problem.rollout(us)

    # Handy to blat up the state and control trajectories
    import matplotlib.pyplot as plt
    crocoddyl.plotOCSolution(xs, us, show=False, figIndex=99, figTitle="Test rollout")
    fig = plt.gcf()
    axs = fig.axes
    for ax in axs:
        ax.grid()

# Now stabilize the acrobot using FDDP
solver = crocoddyl.SolverFDDP(problem)

# Solve
callbacks = []
callbacks.append(crocoddyl.CallbackLogger())
callbacks.append(crocoddyl.CallbackVerbose())
solver.setCallbacks(callbacks)
solver.solve([], [], 300, False, 1e-5)

if WITHDISPLAY:
    # Display from the solver
    display = crocoddyl.GepettoDisplay(robot, floor=False)
    display.displayFromSolver(solver)

if WITHPLOT:
    # Plotting the solution and the DDP convergence
    log = solver.getCallbacks()[0]

    import matplotlib.pyplot as plt

    crocoddyl.plotOCSolution(xs=log.xs, us=log.us, show=False, figIndex=1, figTitle="Solution")
    fig = plt.gcf()
    axs = fig.axes
    for ax in axs:
        ax.grid(True)

    crocoddyl.plotConvergence(log.costs,
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              show=False,
                              figIndex=2)
    fig = plt.gcf()
    axs = fig.axes
    for ax in axs:
        ax.grid(True)

    plt.show()
