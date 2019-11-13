import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

crocoddyl.switchToNumpyMatrix()

# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.

# First, let's load the Pinocchio model for the Talos arm.
# talos_arm = example_robot_data.loadTalosArm()
talos_arm = example_robot_data.loadDoublePendulum()
robot_model = talos_arm.model

# Create a cost model per the running and terminal action model.
state = crocoddyl.StateMultibody(robot_model)
actuationModel = crocoddyl.ActuationModelFull(state)
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

Mref = crocoddyl.FramePlacement(robot_model.getFrameId("gripper_left_joint"),
                                pinocchio.SE3(np.eye(3), np.matrix([[.0], [.0], [.4]])))
goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref)
xRegCost = crocoddyl.CostModelState(state)
uRegCost = crocoddyl.CostModelControl(state)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-3)
runningCostModel.addCost("xReg", xRegCost, 1e-7)
runningCostModel.addCost("uReg", uRegCost, 1e-7)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)

m = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModel)
mND = crocoddyl.DifferentialActionModelNumDiff(m)

d = m.createData()
dND = mND.createData()

for i in range(10):
    x = m.state.rand()
    u = pinocchio.utils.rand(m.nu)
    m.calcDiff(d, x, u)
    mND.calcDiff(dND, x, u)
    diff_err = np.linalg.norm(d.Fx - dND.Fx)
    print(diff_err)

print("finished")
