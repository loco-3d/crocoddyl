from robots import loadTalosArm,loadTalosLegs
from crocoddyl import ActionModelImpact,ImpulseModel6D,ImpulseModelMultiple
from crocoddyl import ActionModelNumDiff
from crocoddyl import m2a, a2m, absmax, absmin

# --- TALOS ARM
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

mnum = ActionModelNumDiff(model,withGaussApprox=True)
dnum = mnum.createData()

nx,ndx,nq,nv,nu = model.nx,model.ndx,model.nq,model.nv,model.nu

mnum.calcDiff(dnum,x,None)
assert( absmax(dnum.Fx-data.Fx) < 1e-3 )
assert( absmax(dnum.Rx-data.Rx) < 1e-3 )
assert( absmax(dnum.Lx-data.Lx) < 1e-3 )
assert( data.Fu.shape[1]==0 and data.Lu.shape == (0,))

# --- TALOS LEGS
robot = loadTalosLegs()
rmodel = robot.model

conactName = 'left_sole_link'

impulse6     = ImpulseModel6D(rmodel,rmodel.getFrameId(contactName))
impulseModel = ImpulseModelMultiple(rmodel,{ "6d": impulse6 })
model        = ActionModelImpact(rmodel,impulseModel)
data         = model.createData()

x = model.State.rand()
q = a2m(x[:model.nq])
v = a2m(x[model.nq:])

model.calc(data,x)
model.calcDiff(data,x)

mnum = ActionModelNumDiff(model,withGaussApprox=True)
dnum = mnum.createData()

nx,ndx,nq,nv,nu = model.nx,model.ndx,model.nq,model.nv,model.nu

mnum.calcDiff(dnum,x,None)

assert( absmax(dnum.Fx-data.Fx) < 1e-3 )
assert( absmax(dnum.Rx-data.Rx) < 1e-3 )
assert( absmax(dnum.Lx-data.Lx) < 1e-3 )
assert( data.Fu.shape[1]==0 and data.Lu.shape == (0,))
