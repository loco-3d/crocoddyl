import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
from double_pendulum_utils import DifferentialActionModelDoublePendulum, CostModelDoublePendulum

robot = example_robot_data.loadDoublePendulum()
robot_model = robot.model

robot.initViewer(loadModel=True)

q0 = np.array([3.14, 0])
robot.q0.flat = q0
robot.framesForwardKinematics(robot.q0)
robot.display(robot.q0)

IDX_LINK1 = robot.model.getFrameId('link1', pinocchio.FrameType.BODY)
IDX_LINK2 = robot.model.getFrameId('link2', pinocchio.FrameType.BODY)
Mlink1 = robot.data.oMf[IDX_LINK1]
Mlink2 = robot.data.oMf[IDX_LINK2]

target_pos = np.array([0, 0, 0.3])
target_quat = pinocchio.Quaternion(1, 0, 0, 0)
target_quat.normalize()

Mref = pinocchio.SE3()
Mref.translation = target_pos.reshape(3, 1)
Mref.rotation = target_quat.matrix()

robot.viewer.gui.refresh()

state = crocoddyl.StateMultibody(robot_model)
runningCostModel = crocoddyl.CostModelSum(state, 1)
terminalCostModel = crocoddyl.CostModelSum(state, 1)

xRegCost = crocoddyl.CostModelState(state, 1)
# xRegCost = CostModelState(robot.model, state, ref=state.zero(), nu=1)
uRegCost = crocoddyl.CostModelControl(state, 1)
# uRegCost = CostModelControl(robot.model, nu = 1)
xPendCost = CostModelDoublePendulum(robot.model,
                                    state,
                                    1,
                                    crocoddyl.ActivationModelWeightedQuad(np.array([1, 1, 1, 1] + [0.1] * 2)))

runningCostModel.addCost("regx", xRegCost, 1e-6)
runningCostModel.addCost("regu", uRegCost, 1e-3)
runningCostModel.addCost("pend", xPendCost, 1)
terminalCostModel.addCost("ori2", xPendCost, 1e5)

actModel = ActuationModelDoublePendulum(robot.model, actLink=2)
# runningModel = IntegratedActionModelEuler(DifferentialActionModelActuated(robot.model, actModel, runningCostModel))
# terminalModel = IntegratedActionModelEuler(DifferentialActionModelActuated(robot.model, actModel, terminalCostModel))
#
# dt = 1e-2
# runningModel.timeStep = dt
#
# T = 100
# x0 = np.array([3.14, 0, 0., 0. ])
# problem = ShootingProblem(x0, [runningModel] * T, terminalModel)
#
# ddp = SolverFDDP(problem)
# ddp.callback = [CallbackDDPVerbose()]
# ddp.callback.append(CallbackDDPLogger())
#
# us0 = np.zeros([T,1])
# xs0 = [problem.initialState+0.1]*len(ddp.models())
#
# ddp.solve(init_xs=xs0,init_us=us0,maxiter=150)
#
# displayTrajectory(robot, ddp.xs, runningModel.timeStep)
