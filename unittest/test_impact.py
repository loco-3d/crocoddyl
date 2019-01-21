import refact
import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,norm,pinv
from numpy import dot,asarray
from numpy.linalg import inv,pinv,norm,svd,eig
from robots import loadTalosArm
from impact import ActionModelImpact,ImpulseModel6D
from continuous import ActionModelNumDiff

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
absmax = lambda A: np.max(abs(A))
absmin = lambda A: np.min(abs(A))

robot = loadTalosArm(freeFloating=True)
rmodel = robot.model

opPointName = 'root_joint'
contactName = 'gripper_left_figertip_1_link'

opPointName,contactName = contactName,opPointName
CONTACTFRAME = rmodel.getFrameId(contactName)
OPPOINTFRAME = rmodel.getFrameId(opPointName)

impulseModel = ImpulseModel6D(rmodel,rmodel.getFrameId(contactName))
model  = ActionModelImpact(rmodel,impulseModel)
data = model.createData()

x = model.State.rand()
q = a2m(x[:model.nq])
v = a2m(x[model.nq:])

model.calc(data,x)
model.calcDiff(data,x)

mnum = ActionModelNumDiff(model)
dnum = mnum.createData()

nx,ndx,nq,nv = model.nx,model.ndx,model.nq,model.nv

mnum.calcDiff(dnum,x,None)
assert( absmax(dnum.Fx[nv:,:nv]-data.Fx[nv:,:nv]) < 1e-3 )
