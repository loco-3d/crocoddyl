import numpy as np

import pinocchio
import example_robot_data

import crocoddyl
import crocoddyl_legacy


class SimpleQuadrotorProblem:
    def __init__(self):
        self.robot = example_robot_data.loadHector()
        self.rmodel = self.robot.model
        self.state = crocoddyl.StateMultibody(self.rmodel)
        d_cog = 0.1525
        cf = 6.6e-5
        cm = 1e-6
        self.tau_f = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0],
                               [0.0, d_cog, 0.0, -d_cog], [-d_cog, 0.0, d_cog, 0.0],
                               [-cm / cf, cm / cf, -cm / cf, cm / cf]])

        self.actuation = crocoddyl.ActuationModelMultiCopterBase(self.state, 4, self.tau_f)
        self.dt = 3e-2
        self.T = 33

        self.runningCostModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        self.terminalCostModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)

        target_pos = np.array([1, 0, 1])
        target_quat = pinocchio.Quaternion(1, 0, 0, 0)
        # Needed objects to create the costs
        Mref = crocoddyl.FramePlacement(self.rmodel.getFrameId("base_link"),
                                        pinocchio.SE3(target_quat.matrix(), target_pos))
        wBasePos, wBaseOri, wBaseVel = 0.1, 1000, 1000
        stateWeights = np.array([wBasePos] * 3 + [wBaseOri] * 3 + [wBaseVel] * self.rmodel.nv)

        # Costs
        goalTrackingCost = crocoddyl.CostModelFramePlacement(self.state, Mref, self.actuation.nu)
        xRegCost = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelWeightedQuad(stateWeights),
                                            self.state.zero(), self.actuation.nu)
        uRegCost = crocoddyl.CostModelControl(self.state, self.actuation.nu)

        self.runningCostModel.addCost("xReg", xRegCost, 1e-6)
        self.runningCostModel.addCost("uReg", uRegCost, 1e-6)
        self.runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
        self.terminalCostModel.addCost("goalPose", goalTrackingCost, 100)

    def createProblemLegacy(self):
        runningModel = crocoddyl_legacy.IntegratedActionModelRK4(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.runningCostModel),
            self.dt)
        terminalModel = crocoddyl_legacy.IntegratedActionModelRK4(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.terminalCostModel), 0.)

        problem = crocoddyl.ShootingProblem(np.concatenate([self.robot.q0, np.zeros(self.state.nv)]),
                                            [runningModel] * self.T, terminalModel)

        return problem

    def createProblem(self):
        runningModel = crocoddyl.IntegratedActionModelRK4(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.runningCostModel),
            self.dt)
        terminalModel = crocoddyl.IntegratedActionModelRK4(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.terminalCostModel), 0.)

        problem = crocoddyl.ShootingProblem(np.concatenate([self.robot.q0, np.zeros(self.state.nv)]),
                                            [runningModel] * self.T, terminalModel)

        return problem


class ArmManipulationProblem:
    def __init__(self):
        self.robot = example_robot_data.loadTalosArm()
        self.rmodel = self.robot.model
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.runningCostModel = crocoddyl.CostModelSum(self.state)
        self.terminalCostModel = crocoddyl.CostModelSum(self.state)

        Mref = crocoddyl.FramePlacement(self.rmodel.getFrameId("gripper_left_joint"),
                                        pinocchio.SE3(np.eye(3), np.array([.0, .0, .4])))
        goalTrackingCost = crocoddyl.CostModelFramePlacement(self.state, Mref)
        xRegCost = crocoddyl.CostModelState(self.state)
        uRegCost = crocoddyl.CostModelControl(self.state)

        self.runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
        self.runningCostModel.addCost("xReg", xRegCost, 1e-4)
        self.runningCostModel.addCost("uReg", uRegCost, 1e-4)
        self.terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)

        self.actuation = crocoddyl.ActuationModelFull(self.state)

        self.dt = 1e-3
        self.T = 250
        self.q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
        self.x0 = np.concatenate([self.q0, pinocchio.utils.zero(self.rmodel.nv)])

    def createProblemLegacy(self):
        runningModel = crocoddyl_legacy.IntegratedActionModelRK4(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.runningCostModel),
            self.dt)
        runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
        terminalModel = crocoddyl_legacy.IntegratedActionModelRK4(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.terminalCostModel), 0.)
        terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

        problem = crocoddyl.ShootingProblem(self.x0, [runningModel] * self.T, terminalModel)

        return problem

    def createProblem(self):
        runningModel = crocoddyl.IntegratedActionModelRK4(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.runningCostModel),
            self.dt)
        runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
        terminalModel = crocoddyl.IntegratedActionModelRK4(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, self.terminalCostModel), 0.)
        terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

        problem = crocoddyl.ShootingProblem(self.x0, [runningModel] * self.T, terminalModel)

        return problem


fly = SimpleQuadrotorProblem()

problem_bindings = fly.createProblem()
problem_legacy = fly.createProblemLegacy()

fddp_bindings = crocoddyl.SolverFDDP(problem_bindings)
fddp_legacy = crocoddyl.SolverFDDP(problem_legacy)

fddp_bindings.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
fddp_legacy.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

fddp_bindings.solve()
fddp_legacy.solve()

for xb, xl in zip(fddp_bindings.xs, fddp_legacy.xs):
    different = False
    if not np.all(np.isclose(xl, xb, atol=1e-10)):
        different = True

if not different:
    print("Quadrotor results are close enough")

manipulate = ArmManipulationProblem()

problem_bindings = manipulate.createProblem()
problem_legacy = manipulate.createProblemLegacy()

fddp_bindings = crocoddyl.SolverFDDP(problem_bindings)
fddp_legacy = crocoddyl.SolverFDDP(problem_legacy)

fddp_bindings.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
fddp_legacy.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

fddp_bindings.solve()
fddp_legacy.solve()

for xb, xl in zip(fddp_bindings.xs, fddp_legacy.xs):
    different = False
    if not np.all(np.isclose(xl, xb, atol=1e-10)):
        different = True

if not different:
    print("Manipulation results are close enough")
