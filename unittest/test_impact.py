from robots import loadTalosArm
from crocoddyl import ActionModelImpact,ImpulseModel6D
from crocoddyl import ActionModelNumDiff
from crocoddyl import m2a, a2m, absmax, absmin

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