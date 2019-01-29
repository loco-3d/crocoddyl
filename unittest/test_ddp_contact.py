import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,norm,pinv
from numpy import dot,asarray
from crocoddyl import IntegratedActionModelEuler, DifferentialActionModelNumDiff,StatePinocchio,CostModelSum,CostModelPinocchio,CostModelPosition,CostModelState,CostModelControl,DifferentialActionModel
from crocoddyl import ContactModel6D,ActuationModelFreeFloating,DifferentialActionModelFloatingInContact,ContactModelMultiple
import warnings
from numpy.linalg import inv,pinv,norm,svd,eig

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
absmax = lambda A: np.max(abs(A))
absmin = lambda A: np.min(abs(A))

from crocoddyl import loadTalosArm
opPointName = 'gripper_left_fingertip_2_link'
contactName = 'root_joint'

opPointName,contactName = contactName,opPointName


class FF:
    def __init__(self):
        robot = self.robot = loadTalosArm(freeFloating=True)
        rmodel = self.rmodel = robot.model
                
        qmin = rmodel.lowerPositionLimit; qmin[:7]=-1; rmodel.lowerPositionLimit = qmin
        qmax = rmodel.upperPositionLimit; qmax[:7]= 1; rmodel.upperPositionLimit = qmax
        State = self.State = StatePinocchio(rmodel)
        actModel = self.actModel = ActuationModelFreeFloating(rmodel)
        contactModel = self.contactModel = ContactModelMultiple(rmodel)
        contact6 = ContactModel6D(rmodel,rmodel.getFrameId(contactName),ref=None)
        contactModel.addContact(name='contact',contact=contact6)
        costModel = self.costModel = CostModelSum(rmodel,nu=actModel.nu)
        self.cost1 = CostModelPosition(rmodel,nu=actModel.nu,
                                     frame=rmodel.getFrameId(opPointName),
                                     ref=np.array([.5,.4,.3]))
        self.cost2 = CostModelState(rmodel,State,ref=State.zero(),nu=actModel.nu)
        self.cost3 = CostModelControl(rmodel,nu=actModel.nu)
        costModel.addCost( name="pos", weight = 10, cost = self.cost1)
        costModel.addCost( name="regx", weight = 0.1, cost = self.cost2) 
        costModel.addCost( name="regu", weight = 0.01, cost = self.cost3)

        self.dmodel = DifferentialActionModelFloatingInContact(rmodel,actModel,contactModel,costModel)
        self.model  = IntegratedActionModelEuler(self.dmodel)
        self.data = self.model.createData()

        self.cd1 = self.data.differential.costs .costs['pos']
        self.cd2 = self.data.differential.costs .costs['regx']
        self.cd3 = self.data.differential.costs .costs['regu']

        self.ddata = self.data.differential
        self.rdata = self.data.differential.pinocchio
        
        self.x = self.State.rand()
        self.q = a2m(self.x[:rmodel.nq])
        self.v = a2m(self.x[rmodel.nq:])
        self.u = np.random.rand(self.model.nu)
    def calc(self,x=None,u=None):
        return self.model.calc(self.data,x if x is not None else self.x,u if u is not None else self.u)
       
class Fix:
    def __init__(self):
        robot = self.robot = loadTalosArm()
        rmodel = self.rmodel = robot.model
        rmodel.armature = np.matrix([ 0 ]*robot.model.nv).T
        for j in robot.model.joints[1:]:
            if j.shortname()!='JointModelFreeFlyer':
                rmodel.armature[j.idx_v:j.idx_v+j.nv]=1
        State = self.State = StatePinocchio(rmodel)
        self.cost1 = CostModelPosition(rmodel,nu=rmodel.nv,
                                       frame=rmodel.getFrameId(opPointName),
                                       ref=np.array([.5,.4,.3]))
        self.cost2 = CostModelState(rmodel,State,ref=State.zero(),nu=rmodel.nv)
        self.cost3 = CostModelControl(rmodel,nu=rmodel.nv)

        self.dmodel = DifferentialActionModel(rmodel)
        self.model  = IntegratedActionModelEuler(self.dmodel)

        self.costModel = costModel = self.dmodel.costs
        costModel.addCost( name="pos", weight = 10, cost = self.cost1)
        costModel.addCost( name="regx", weight = 0.1, cost = self.cost2) 
        costModel.addCost( name="regu", weight = 0.01, cost = self.cost3)

        self.data = self.model.createData()
        
        self.cd1 = self.data.differential.costs .costs['pos']
        self.cd2 = self.data.differential.costs .costs['regx']
        self.cd3 = self.data.differential.costs .costs['regu']

        self.ddata = self.data.differential
        self.rdata = self.data.differential.pinocchio

        self.x = self.State.rand()
        self.q = a2m(self.x[:rmodel.nq])
        self.v = a2m(self.x[rmodel.nq:])
        self.u = np.random.rand(self.model.nu)
    def calc(self,x=None,u=None):
        return self.model.calc(self.data,x if x is not None else self.x,u if u is not None else self.u)
        
ff = FF()
fix = Fix()

ff.q[:7].flat = ff.State.zero()[:7].flat
ff.v[:6] = 0
fix.q[:] = ff.q[7:]
fix.v[:] = ff.v[6:]
fix.u[:] = ff.u[:]
fix.x[:] = np.concatenate([fix.q,fix.v]).flat
ff.x[:] = np.concatenate([ff.q,ff.v]).flat

ff.model.timeStep = fix.model.timeStep = 5e-3

xfix,cfix = fix.model.calc(fix.data,fix.x,fix.u)
xff, cff  = ff.model.calc(ff.data,ff.x,ff.u)

if ff.contactModel['contact'].frame == 1:
    assert(norm(cff-cfix)<1e-6)
    assert(norm(xff[7:ff.rmodel.nq]-xfix[:fix.rmodel.nq])<1e-6)
    assert(norm(xff[ff.rmodel.nq+6:]-xfix[fix.rmodel.nq:])<1e-6)

# --- DDP 
# --- DDP 
# --- DDP 
from crocoddyl import ShootingProblem, SolverDDP,SolverKKT

ff.model.timeStep = fix.model.timeStep = 1e-2
T = 20

xref = fix.State.rand(); xref[fix.rmodel.nq:] = 0
fix.calc(xref)
#ref = fix.rdata.oMf[fix.cost1.frame].translation.flat
ref = [0.5, 0.4, 0.3]

f = ff
f.u[:] = (0*pinocchio.rnea(fix.rmodel,fix.rdata,fix.q,fix.v*0,fix.v*0)).flat
f.v[:] = 0
f.x[f.rmodel.nq:] = f.v.flat
f.cost1.ref[:] = ref

#f.u[:] = np.zeros(f.model.nu)
f.model.differential.costs['pos' ].weight = 1
f.model.differential.costs['regx'].weight = 0.01
f.model.differential.costs['regu'].weight = 0.0001

fterm = f.__class__()
fterm.model.differential.costs['pos' ].weight = 1000
fterm.model.differential.costs['regx'].weight = 1
fterm.model.differential.costs['regu'].weight = 0.01
fterm.cost1.ref[:] = ref
fterm.model.differential.costs['regx'].cost.weights = np.array([0]*6+[0.01]*(f.rmodel.nv-6)+[10]*f.rmodel.nv)


problem = ShootingProblem(f.x, [ f.model ]*T, fterm.model)
u0s = [ f.u ]*T
x0s = problem.rollout(u0s)

from crocoddyl import CallbackDDPLogger
disp = lambda xs: disptraj(f.robot,xs)
ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPLogger()]
ddp.th_stop = 1e-18
ddp.solve(verbose=True,maxiter=1000)
