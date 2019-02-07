from crocoddyl import m2a, a2m, absmax, absmin
import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,pinv,norm,svd,eig


## Loading Talos arm with FF TODO use a bided or quadruped
# -----------------------------------------------------------------------------
from crocoddyl import loadTalosArm

robot = loadTalosArm(freeFloating=True)
robot.model.armature[6:] = 1.
qmin = robot.model.lowerPositionLimit; qmin[:7]=-1; robot.model.lowerPositionLimit = qmin
qmax = robot.model.upperPositionLimit; qmax[:7]= 1; robot.model.upperPositionLimit = qmax

rmodel = robot.model
rdata = rmodel.createData()



# -----------------------------------------------------------------------------
from crocoddyl import ContactData6D, ContactModel6D
        
contactModel = ContactModel6D(rmodel,rmodel.getFrameId('gripper_left_fingertip_2_link'),
                              ref=pinocchio.SE3.Random(), gains=[4.,4.])
contactData  = contactModel.createData(rdata)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv-6))

pinocchio.forwardKinematics(rmodel,rdata,q,v,zero(rmodel.nv))
pinocchio.computeJointJacobians(rmodel,rdata)
pinocchio.updateFramePlacements(rmodel,rdata)
contactModel.calc(contactData,x)

rdata2 = rmodel.createData()
pinocchio.computeAllTerms(rmodel,rdata2,q,v)
pinocchio.updateFramePlacements(rmodel,rdata2)
contactData2  = contactModel.createData(rdata2)
contactModel.calc(contactData2,x)
assert(norm(contactData.a0-contactData2.a0)<1e-9)
assert(norm(contactData.J -contactData2.J )<1e-9)



# ----------------------------------------------------------------------------
from crocoddyl import ContactData3D, ContactModel3D

contactModel = ContactModel3D(rmodel,
                              rmodel.getFrameId('gripper_left_fingertip_2_link'),ref=None)
contactData  = contactModel.createData(rdata)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv-6))

pinocchio.forwardKinematics(rmodel,rdata,q,v,zero(rmodel.nv))
pinocchio.computeJointJacobians(rmodel,rdata)
pinocchio.updateFramePlacements(rmodel,rdata)
contactModel.calc(contactData,x)

rdata2 = rmodel.createData()
pinocchio.computeAllTerms(rmodel,rdata2,q,v)
pinocchio.updateFramePlacements(rmodel,rdata2)
contactData2  = contactModel.createData(rdata2)
contactModel.calc(contactData2,x)
assert(norm(contactData.a0-contactData2.a0)<1e-9)
assert(norm(contactData.J -contactData2.J )<1e-9)



#---------------------------------------------------------------------
# Many contact model
from crocoddyl import CostModelSum
from crocoddyl import ContactModelMultiple
from crocoddyl import DifferentialActionModelFloatingInContact
from crocoddyl import DifferentialActionModelNumDiff
from crocoddyl import ActuationModelFreeFloating



q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)*2-1

pinocchio.computeJointJacobians(rmodel,rdata,q)
J6 = pinocchio.getJointJacobian(rmodel,rdata,rmodel.joints[-1].id,
                                pinocchio.ReferenceFrame.LOCAL).copy()
J = J6[:3,:]
v -= pinv(J)*J*v

x = np.concatenate([ m2a(q),m2a(v) ])
u = np.random.rand(rmodel.nv-6)*2-1

actModel = ActuationModelFreeFloating(rmodel)
contactModel3 = ContactModel3D(rmodel,rmodel.getFrameId('gripper_left_fingertip_2_link'),
                               ref=None)
rmodel.frames[contactModel3.frame].placement = pinocchio.SE3.Random()
contactModel = ContactModelMultiple(rmodel)
contactModel.addContact(name='fingertip',contact=contactModel3)

model = DifferentialActionModelFloatingInContact(rmodel,actModel,
                                                 contactModel,CostModelSum(rmodel))
data  = model.createData()

model.calc(data,x,u)
assert( len(filter(lambda x:x>0,eig(data.K)[0])) == model.nv )
assert( len(filter(lambda x:x<0,eig(data.K)[0])) == model.ncontact )
_taucheck = pinocchio.rnea(rmodel,rdata,q,v,a2m(data.a),data.contact.forces)
_taucheck.flat[:] += rmodel.armature.flat*data.a
assert( absmax(_taucheck[:6])<1e-6 )
assert( absmax(m2a(_taucheck[6:])-u)<1e-6 )

model.calcDiff(data,x,u)

mnum = DifferentialActionModelNumDiff(model,withGaussApprox=False)
dnum = mnum.createData()
mnum.calcDiff(dnum,x,u)
assert(absmax(data.Fx-dnum.Fx)/model.nx<1e-3)
assert(absmax(data.Fu-dnum.Fu)/model.nu<1e-3)


#------------------------------------------------
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)*2-1
x = np.concatenate([ m2a(q),m2a(v) ])
u = np.random.rand(rmodel.nv-6)*2-1

actModel = ActuationModelFreeFloating(rmodel)
contactModel6 = ContactModel6D(rmodel,rmodel.getFrameId('gripper_left_fingertip_2_link'),
                               ref=pinocchio.SE3.Random(), gains=[4.,4.])
rmodel.frames[contactModel6.frame].placement = pinocchio.SE3.Random()
contactModel = ContactModelMultiple(rmodel)
contactModel.addContact(name='fingertip',contact=contactModel6)

model = DifferentialActionModelFloatingInContact(rmodel,actModel,contactModel,CostModelSum(rmodel))
data  = model.createData()

model.calc(data,x,u)
assert( len(filter(lambda x:x>0,eig(data.K)[0])) == model.nv )
assert( len(filter(lambda x:x<0,eig(data.K)[0])) == model.ncontact )
_taucheck = pinocchio.rnea(rmodel,rdata,q,v,a2m(data.a),data.contact.forces)
if hasattr(rmodel,'armature'): _taucheck.flat += rmodel.armature.flat*data.a
assert( absmax(_taucheck[:6])<1e-6 )
assert( absmax(m2a(_taucheck[6:])-u)<1e-6 )

model.calcDiff(data,x,u)

mnum = DifferentialActionModelNumDiff(model,withGaussApprox=False)
dnum = mnum.createData()
mnum.calcDiff(dnum,x,u)
assert(absmax(data.Fx-dnum.Fx)/model.nx<1e-3)
assert(absmax(data.Fu-dnum.Fu)/model.nu<1e-3)



#----------------------------------------------------------
### Check force derivatives
def df_dq(model,func,q,h=1e-9):
    dq = zero(model.nv)
    f0 = func(q)
    res = zero([len(f0),model.nv])
    for iq in range(model.nv):
        dq[iq] = h
        res[:,iq] = (func(pinocchio.integrate(model,q,dq)) - f0)/h
        dq[iq] = 0
    return res

def df_dv(model,func,v,h=1e-9):
    dv = zero(model.nv)
    f0 = func(v)
    res = zero([len(f0),model.nv])
    for iv in range(model.nv):
        dv[iv] = h
        res[:,iv] = (func(v+dv) - f0)/h
        dv[iv] = 0
    return res

def df_dz(model,func,z,h=1e-9):
    dz = zero(len(z))
    f0 = func(z)
    res = zero([len(f0),len(z)])
    for iz in range(len(z)):
        dz[iz] = h
        res[:,iz] = (func(z+dz) - f0)/h
        dz[iz] = 0
    return res

def calcForces(q_,v_,u_):
    model.calc(data,np.concatenate([m2a(q_),m2a(v_)]),m2a(u_))
    return a2m(data.f)

Fq = df_dq(rmodel,lambda _q: calcForces(_q,v,u), q)
Fv = df_dv(rmodel,lambda _v: calcForces(q,_v,u), v)
Fu = df_dz(rmodel,lambda _u: calcForces(q,v,_u), a2m(u))
assert( absmax(Fq-data.df_dq) < 1e-3 )
assert( absmax(Fv-data.df_dv) < 1e-3 )
assert( absmax(Fu-data.df_du) < 1e-3 )


# -------------------------------------------------------------------------------
# Cost force model
from crocoddyl import CostDataForce, CostModelForce

model.costs = CostModelSum(rmodel,nu=actModel.nu)
model.costs.addCost( name='force', weight = 1,
                    cost = CostModelForce(rmodel,model.contact.contacts['fingertip'],
                                            nu=actModel.nu) )

data = model.createData()
data.costs['force'].contact = data.contact[model.costs['force'].cost.contact]

cmodel = model.costs['force'].cost
cdata  = data .costs['force'] 

model.calcDiff(data,x,u)

mnum = DifferentialActionModelNumDiff(model,withGaussApprox=False)
dnum = mnum.createData()
for d in dnum.datax:    d.costs['force'].contact = d.contact[model.costs['force'].cost.contact]
for d in dnum.datau:    d.costs['force'].contact = d.contact[model.costs['force'].cost.contact]
dnum.data0.costs['force'].contact = dnum.data0.contact[model.costs['force'].cost.contact]
    

mnum.calcDiff(dnum,x,u)
assert(absmax(data.Fx-dnum.Fx)/model.nx<1e-3)
assert(absmax(data.Fu-dnum.Fu)/model.nu<1e-3)




# TODO Check if we need this unit-test here. Note that is an ction test
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# --- COMPLETE MODEL WITH COST ----
from crocoddyl import StatePinocchio
from crocoddyl import CostModelFrameTranslation, CostModelState, CostModelControl
from crocoddyl import IntegratedActionModelEuler
State = StatePinocchio(rmodel)

actModel = ActuationModelFreeFloating(rmodel)
contactModel = ContactModelMultiple(rmodel)
contactModel.addContact(name='root_joint',contact=contactModel6)

costModel = CostModelSum(rmodel,nu=actModel.nu)
costModel.addCost( name="pos", weight = 10,
                   cost = CostModelFrameTranslation(rmodel,nu=actModel.nu,
                                            frame=rmodel.getFrameId('gripper_left_inner_single_link'),
                                            ref=np.array([.5,.4,.3])))
costModel.addCost( name="regx", weight = 0.1,
                   cost = CostModelState(rmodel,State,ref=State.zero(),nu=actModel.nu) )
costModel.addCost( name="regu", weight = 0.01,
                   cost = CostModelControl(rmodel,nu=actModel.nu) )

c1 = costModel.costs['pos'].cost
c2 = costModel.costs['regx'].cost
c3 = costModel.costs['regu'].cost

dmodel = DifferentialActionModelFloatingInContact(rmodel,actModel,contactModel,costModel)
model  = IntegratedActionModelEuler(dmodel)
data = model.createData()

d1 = data.differential.costs .costs['pos']
d2 = data.differential.costs .costs['regx']
d3 = data.differential.costs .costs['regu']

from crocoddyl import ActionModelNumDiff
mnum = ActionModelNumDiff(model,withGaussApprox=True)
dnum = mnum.createData()

model.calc(data,x,u)
model.calcDiff(data,x,u)

mnum.calcDiff(dnum,x,u)
assert( norm(data.Lx-dnum.Lx) < 1e-3 )
assert( norm(data.Lu-dnum.Lu) < 1e-3 )
assert( norm(dnum.Lxx-data.Lxx) < 1e-3)
assert( norm(dnum.Lxu-data.Lxu) < 1e-3)
assert( norm(dnum.Luu-data.Luu) < 1e-3)









# TODO move to an integrative test
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# integrative test: checking 1-step DDP versus KKT

from crocoddyl import ShootingProblem, SolverDDP,SolverKKT
from crocoddyl import IntegratedActionModelEuler


def disp(xs,dt=0.1):
    if not hasattr(robot,'viewer'): robot.initDisplay(loadModel=True)
    import time
    for x in xs:
        robot.display(a2m(x[:robot.nq]))
        time.sleep(dt)

import copy
from crocoddyl import CallbackDDPLogger

model.timeStep = 1e-1
dmodel.costs['pos'].weight = 1
dmodel.costs['regx'].weight = 0
dmodel.costs['regu'].weight = 0

# Choose a cost that is reachable.
x0 = model.State.rand()
xref = model.State.rand()
xref[:7] = x0[:7]
pinocchio.forwardKinematics(rmodel,rdata,a2m(xref))
pinocchio.updateFramePlacements(rmodel,rdata)
c1.ref[:] = m2a(rdata.oMf[c1.frame].translation.copy())

problem = ShootingProblem(x0, [ model ], model)

ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPLogger()]
ddp.th_stop = 1e-18
xddp,uddp,doneddp = ddp.solve(maxiter=30)

assert(doneddp)
assert( norm(ddp.datas()[-1].differential.costs['pos'].residuals)<1e-3 )
assert( norm(m2a(ddp.datas()[-1].differential.costs['pos'].pinocchio.oMf[c1.frame].translation)\
             -c1.ref)<1e-3 )

u0 = np.zeros(model.nu)
x1 = model.calc(data,problem.initialState,u0)[0]
x0s = [ problem.initialState.copy(), x1 ]
u0s = [ u0.copy() ]

dmodel.costs['regu'].weight = 1e-3

kkt = SolverKKT(problem)
kkt.th_stop = 1e-18
xkkt,ukkt,donekkt = kkt.solve(init_xs=x0s,init_us=u0s,isFeasible=True,maxiter=20)
xddp,uddp,doneddp = ddp.solve(init_xs=x0s,init_us=u0s,isFeasible=True,maxiter=20)

assert(donekkt)
assert(norm(xkkt[0]-problem.initialState)<1e-9)
assert(norm(xddp[0]-problem.initialState)<1e-9)
for t in range(problem.T):
    assert(norm(ukkt[t]-uddp[t])<1e-6)
    assert(norm(xkkt[t+1]-xddp[t+1])<1e-6)
