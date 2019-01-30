from crocoddyl import DifferentialActionModel, IntegratedActionModelEuler
from crocoddyl import ShootingProblem
from crocoddyl import CostModelFramePlacement, CostModelState, CostModelControl, CostModelFrameTranslation
from crocoddyl import StatePinocchio, a2m, m2a
import pinocchio
import numpy as np


class PointToPointProblem:
    def __init__(self, robot, frameId):
        self.robot = robot
        self.frameId = frameId
        self.state = StatePinocchio(self.robot.model)
    def createProblem(self, x0, T, dt, p1, p2, p3, p4):
        """ Define 4 points picking points
        """
        # Our robot needs to pass through 4 points and then come back to the default posture
        # First we defined the running action models per each point
        
        
        q0 = np.asmatrix(x0[:self.robot.model.nq]).T
        p0 = self.robot.framePlacement(q0, self.frameId).translation

        ######################################################################
        ######### TODO: Add the cost functions per each action model #########
        ######################################################################
        # Hints: Indenpendely of the action, you need to define three cost functions:
        # goToCost, uRegCost and xRegCost. These are the main points:
        #  1. Define the uRegCost and xRegCost.
        #  2. Add these cost function in each action model (e.g. goFirstPointAction)
        #     with weights: 1./1e-3 (terminal/running goToCost), 1e-7 (uRegCost and xRegCost)
        # For adding the cost functions you should do:
        #  myAction.differential.costs.addCost( name="myCostName", weight = myWeight, cost = myCostObject)
        # runningCostModel = runningModel.differential.costs
        goFirstPointModels = \
            [ self.createPointToPointModel(
                dt,
                p0 + (p1 - p0) * k/T
                ) for k in range(T) ]
        firstPointModel = self.createTerminalPointModel(p1)

        goSecondPointModels = \
            [ self.createPointToPointModel(
                dt,
                p1 + (p2 - p1) * k/T
                ) for k in range(T) ]
        secondPointModel = self.createTerminalPointModel(p2)

        goThirdPointModels = \
            [ self.createPointToPointModel(
                dt,
                p2 + (p3 - p2) * k/T
                ) for k in range(T) ]
        thirdPointModel = self.createTerminalPointModel(p3)

        # Define the set of action models (we call it taskModels)
        # goHomeModels = [ goHomeAction ] * T + [ goFourthPointActionTerm ]
        taskModels = goFirstPointModels + goSecondPointModels # + goThirdPointModels

        # taskModels = goFirstPointModels + [ firstPointModel ]  \
        #     + goSecondPointModels + [ secondPointModel ] \
        #     + goThirdPointModels + [ thirdPointModel ] #goFourthPointModels# + goHomeModels


        # Building a shooting problem from a stack of action models
        problem = ShootingProblem(x0, taskModels, goThirdPointModels[-1])
        return problem
    def createPointToPointModel(self, dt, p):
        pointToPointAction = IntegratedActionModelEuler(DifferentialActionModel(self.robot.model))
        
        # Setting up the time step per each action model
        pointToPointAction.timeStep = dt

        goToWeight = 0.5 #1e-1
        uRegWeight = 1e-5
        xRegWeight = 1e-5
        pointToPointAction.differential.costs.addCost(
            name="goTo", weight = goToWeight, cost = self.goToCost(p))
        pointToPointAction.differential.costs.addCost(
            name="uReg", weight = uRegWeight, cost = self.uRegCost())
        pointToPointAction.differential.costs.addCost(
            name="xReg", weight = xRegWeight, cost = self.xRegCost())
        return pointToPointAction

    def createTerminalPointModel(self, p):
        terminalPointAction = IntegratedActionModelEuler(DifferentialActionModel(self.robot.model))
        
        goToWeight = 10.
        terminalPointAction.differential.costs.addCost(
            name="goTo", weight = goToWeight, cost = self.goToCost(p))
        return terminalPointAction

    def goToCost(self, p_ref):
        # Return SE3 tracking cost
        SE3ref = pinocchio.SE3(np.eye(3), p_ref)
        return CostModelFramePlacement(self.robot.model,
                                   nu=self.robot.model.nv,
                                   frame=self.frameId,
                                   ref=SE3ref)#,
                                #    weights=[1.,1.,1.,0.2,0.2,0.2])
        # return CostModelPosition(self.robot.model,
        #                            nu=self.robot.model.nv,
        #                            frame=self.frameId,
        #                            ref=m2a(p_ref))

    def uRegCost(self):
        # Return the control regularizatin cost
        return CostModelControl(self.robot.model,nu=self.robot.model.nv)
    def xRegCost(self):
        # Return the state regularization cost
        return CostModelState(self.robot.model,
                              self.state,
                              ref=self.state.zero(),
                              nu=self.robot.model.nv)



from crocoddyl import SolverDDP
from crocoddyl import CallbackDDPLogger, CallbackDDPVerbose, CallbackSolverDisplay
from crocoddyl import loadTalosArm


# Loading the Talos arm and defining the wripper frame
talos_arm = loadTalosArm()
wripperFrame = talos_arm.model.getFrameId('gripper_left_joint')

# Defining a initial state, Horizon and time step
q0 = [0.173046, 1., -0.52366, 0., 0., 0.1, -0.005]
x0 = np.hstack([q0, np.zeros(talos_arm.model.nv)])
T = 250
dt = 1e-3



# Creating the point-to-point problem
p2p = PointToPointProblem(talos_arm, wripperFrame)
p1 = a2m([0., 0.45, 0.])
p2 = a2m([0.2, 0.45, 0.])
p3 = a2m([0.2, -0.2, 0.3])
p4 = a2m([0.2, -0.2, 0.2])
problem = p2p.createProblem(x0, T, dt, p1, p2, p3, p4)

# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose(1), CallbackSolverDisplay(talos_arm,4,cameraTF)]

# Solving it with the DDP algorithm
ddp.solve(maxiter=200)


# Visualizing the solution in gepetto-viewer
CallbackSolverDisplay(talos_arm)(ddp)



# Printing the reached position
log = ddp.callback[0]
xT = log.xs[-1]
qT = np.asmatrix(xT[:talos_arm.model.nq]).T
print
print "The reached pose by the wrist is"
print talos_arm.framePlacement(qT, wripperFrame)