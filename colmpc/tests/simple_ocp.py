import coal
import crocoddyl
import matplotlib.pyplot as plt
import mim_solvers
import numpy as np
import pinocchio as pin
from numpy import r_
from wrapper_panda import PandaWrapper

import colmpc as col

np.set_printoptions(precision=4, linewidth=350, suppress=True, threshold=1e6)

# from VelocityResidual import ResidualModelVelocityAvoidance
# from colmpc import ResidualModelVelocityAvoidance


### PARAMETERS
# Number of nodes of the trajectory
T = 20
# Time step between each node
dt = 0.1

INITIAL_CONFIG = np.array(
    [
        -0.06709294,
        1.35980773,
        -0.81605989,
        0.74243348,
        0.42419277,
        0.45547585,
        -0.00456262,
    ]
)
INITIAL_VELOCITY = np.zeros(7)
x0 = np.r_[INITIAL_CONFIG, INITIAL_VELOCITY]
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0.5, 1.2]))

ksi = 1e-1
di = 1e-1  # 1e-4
ds = 1e-7

WEIGHT_uREG = 1e-4
WEIGHT_xREG = 1e-1
WEIGHT_GRIPPER_POSE = 5
WEIGHT_GRIPPER_POSE_TERM = 100
WEIGHT_LIMIT = 1e-1
SAFETY_THRESHOLD = 0

#### Creating the robot
robot_wrapper = PandaWrapper()
rmodel, gmodel, vmodel = robot_wrapper()

gmodel.removeGeometryObject("panda1_box_0")
vmodel.removeGeometryObject("panda1_box_0")

D1 = np.diagflat([1 / 0.1**2, 1 / 0.2**2, 1 / 0.1**2])
D2 = np.diagflat([1 / 0.04**2, 1 / 0.04**2, 1 / 0.04**2])


Mobs = pin.SE3(
    pin.utils.rotate("y", np.pi) @ pin.utils.rotate("z", np.pi / 2),
    np.array([0, 0.1, 1.2]),
)
rmodel.addFrame(pin.Frame("obstacle", 0, 0, Mobs, pin.OP_FRAME))

idf1 = rmodel.getFrameId("obstacle")
idj1 = rmodel.frames[idf1].parentJoint
elips1 = coal.Ellipsoid(*[d**-0.5 for d in np.diag(D1)])
elips1_geom = pin.GeometryObject(
    "el1", idj1, idf1, rmodel.frames[idf1].placement, elips1
)
elips1_geom.meshColor = r_[1, 0, 0, 1]
idg1 = gmodel.addGeometryObject(elips1_geom)

idf2 = rmodel.getFrameId("panda2_hand_tcp")
idj2 = rmodel.frames[idf2].parentJoint
elips2 = coal.Ellipsoid(*[d**-0.5 for d in np.diag(D2)])
elips2_geom = pin.GeometryObject(
    "el2", idj2, idf2, rmodel.frames[idf2].placement, elips2
)
elips2_geom.meshColor = r_[1, 1, 0, 1]
idg2 = gmodel.addGeometryObject(elips2_geom)

rdata, gdata = rmodel.createData(), gmodel.createData()

gmodel.addCollisionPair(
    pin.CollisionPair(
        gmodel.getGeometryId("el1"),
        gmodel.getGeometryId("el2"),
    )
)

#### Creating the OCP

# Stat and actuation model
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFull(state)

# Running & terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

### Creation of cost terms

# State Regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Control Regularization cost
uResidual = crocoddyl.ResidualModelControl(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# End effector frame cost
framePlacementResidual = crocoddyl.ResidualModelFrameTranslation(
    state,
    rmodel.getFrameId("panda2_rightfinger"),
    TARGET_POSE.translation,
)

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

# Obstacle cost with hard constraint
runningConstraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
terminalConstraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
# Creating the residual

obstacleDistanceResidual_col = col.ResidualModelVelocityAvoidance(
    state, gmodel, 0, di, ds, ksi
)

# Creating the inequality constraint
constraint = crocoddyl.ConstraintModelResidual(
    state,
    obstacleDistanceResidual_col,
    np.array([0]),
    np.array([np.inf]),
)

# Adding the constraint to the constraint manager
runningConstraintModelManager.addConstraint("col_mpc" + str(0), constraint)
terminalConstraintModelManager.addConstraint("col_term_mpc" + str(0), constraint)


# obstacleDistanceResidual = ResidualModelVelocityAvoidance(
# state ,gmodel, idg1, idg2,ksi, di, ds,
# )

# # Creating the inequality constraint
# constraint = crocoddyl.ConstraintModelResidual(
#     state,
#     obstacleDistanceResidual,
#     np.array([0]),
#     np.array([np.inf]),
# )

# # Adding the constraint to the constraint manager
# runningConstraintModelManager.addConstraint(
#     "col_" + str(0), constraint
# )
# terminalConstraintModelManager.addConstraint(
#     "col_term_" + str(0), constraint
# )

# Adding costs to the models
runningCostModel.addCost("stateReg", xRegCost, WEIGHT_xREG)
runningCostModel.addCost("ctrlRegGrav", uRegCost, WEIGHT_uREG)
runningCostModel.addCost("gripperPoseRM", goalTrackingCost, WEIGHT_GRIPPER_POSE)
terminalCostModel.addCost("stateReg", xRegCost, WEIGHT_xREG)
terminalCostModel.addCost("gripperPose", goalTrackingCost, WEIGHT_GRIPPER_POSE_TERM)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
    state,
    actuation,
    runningCostModel,
    runningConstraintModelManager,
)
terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
    state,
    actuation,
    terminalCostModel,
    terminalConstraintModelManager,
)

runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)

runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
# Create solver + callbacks
# Define mim solver with inequalities constraints
ddp = mim_solvers.SolverCSQP(problem)

# Merit function
ddp.use_filter_line_search = False

# Parameters of the solver
ddp.termination_tolerance = 1e-3
ddp.max_qp_iters = 10000
ddp.eps_abs = 1e-6
ddp.eps_rel = 0

ddp.with_callbacks = True
XS_init = [x0] * (T + 1)
US_init = ddp.problem.quasiStatic(XS_init[:-1])


ddp.solve(XS_init, US_init, 100)


# while True:
# for i,xs in enumerate(ddp.xs):
# q = np.array(xs[:7].tolist())
# pin.framesForwardKinematics(rmodel, rdata, q)
# add_cube_to_viewer(viz, "vcolmpc" + str(i), [2e-2,2e-2, 2e-2],
# rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation, color=100000000)
# viz.display(np.array(xs[:7].tolist()))
# input()
d = []
for i, xs in enumerate(ddp.xs):
    q = np.array(xs[:7].tolist())
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)

    req = coal.DistanceRequest()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    res = coal.DistanceResult()
    distance = coal.distance(
        elips1,
        gdata.oMg[idg1],
        elips2,
        gdata.oMg[idg2],
        req,
        res,
    )
    d.append(distance)


plt.plot(d)
plt.plot([0] * len(d))
plt.show()
