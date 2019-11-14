import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
from crocoddyl.utils.quadrotor import ActuationModelUAM

crocoddyl.switchToNumpyMatrix()

hector = example_robot_data.loadHector()
robot_model = hector.model

hector.initViewer(loadModel=True)
hector.display(hector.q0)

target_pos = np.array([1, 0, 1])
target_quat = pinocchio.Quaternion(1, 0, 0, 0)
target_quat.normalize()
hector.viewer.gui.addXYZaxis('world/wp', [1., 0., 0., 1.], .03, 0.5)
hector.viewer.gui.applyConfiguration(
    'world/wp',
    target_pos.tolist() + [target_quat[0], target_quat[1], target_quat[2], target_quat[3]])
hector.viewer.gui.refresh()

state = crocoddyl.StateMultibody(robot_model)
distance_rotor_COG = 0.1525
cf = 6.6e-5
cm = 1e-6
u_lim = 5
l_lim = 0.1
actModel = ActuationModelUAM(state, '+', distance_rotor_COG, cm, cf, u_lim, l_lim)

runningCostModel = crocoddyl.CostModelSum(state, actModel.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)

# Needed objects to create the costs
Mref = crocoddyl.FramePlacement(robot_model.getFrameId("base_link"),
                                pinocchio.SE3(target_quat.matrix(),
                                              np.matrix(target_pos).T))
wBasePos, wBaseOri, wBaseVel, wBaseRate = [0.1], [1000], [1000], [10]
stateWeights = np.matrix([wBasePos * 3 + wBaseOri * 3 + wBaseVel * robot_model.nv]).T

# Costs
goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref, actModel.nu)
xRegCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights), state.zero(),
                                    actModel.nu)
uRegCost = crocoddyl.CostModelControl(state, actModel.nu)

runningCostModel.addCost("regx", xRegCost, 1e-6)
runningCostModel.addCost("regu", uRegCost, 1e-6)
runningCostModel.addCost("pos", goalTrackingCost, 1e-2)
terminalCostModel.addCost("pos", goalTrackingCost, 100)

dt, T = 3e-2, 33

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, terminalCostModel), dt)

problem = crocoddyl.ShootingProblem(np.vstack([hector.q0, pinocchio.utils.zero((state.nv))]), [runningModel]*T, terminalModel)

fddp = crocoddyl.SolverFDDP(problem)

fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

fddp.solve()

crocoddyl.displayTrajectory(hector, fddp.xs, dt)