from crocoddyl import DifferentialActionModelNumDiff
from crocoddyl import m2a, a2m, absmax, absmin
import numpy as np
import pinocchio
from pinocchio.utils import *



## Loading Talos arm with FF TODO use a bided or quadruped
# -----------------------------------------------------------------------------
from crocoddyl import loadTalosArm

robot = loadTalosArm(freeFloating=True)

qmin = robot.model.lowerPositionLimit; qmin[:7]=-1; robot.model.lowerPositionLimit = qmin
qmax = robot.model.upperPositionLimit; qmax[:7]= 1; robot.model.upperPositionLimit = qmax

rmodel = robot.model
rdata = rmodel.createData()




## Free floating actuation model
# -----------------------------------------------------------------------------
from crocoddyl import ActuationModelFreeFloating
from crocoddyl import DifferentialActionModelActuated

actModel = ActuationModelFreeFloating(rmodel)
model = DifferentialActionModelActuated(rmodel,actModel)
data  = model.createData()

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv-6))
model.calcDiff(data,x,u)

mnum = DifferentialActionModelNumDiff(model)
dnum = mnum.createData()
mnum.calcDiff(dnum,x,u)

assert(absmax(data.Fx-dnum.Fx)/model.nx < 1e3*mnum.disturbance )
assert(absmax(data.Fu-dnum.Fu)/model.nu < 1e2*mnum.disturbance )