import refact
import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,norm,pinv
from numpy import dot,asarray
from continuous import DifferentialActionModelPositioning, DifferentialActionModel, IntegratedActionModelEuler, DifferentialActionModelNumDiff,StatePinocchio,CostModelSum,CostModelPinocchio,CostModelPosition,CostModelState,CostModelControl
import warnings
from numpy.linalg import inv,pinv,norm,svd,eig

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
absmax = lambda A: np.max(abs(A))
absmin = lambda A: np.min(abs(A))

path = '/home/nmansard/src/cddp/examples/'

urdf = path + 'talos_data/robots/talos_left_arm.urdf'
robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(urdf, [path] \
                                                           ,pinocchio.JointModelFreeFlyer() \
)

qmin = robot.model.lowerPositionLimit; qmin[:7]=-1; robot.model.lowerPositionLimit = qmin
qmax = robot.model.upperPositionLimit; qmax[:7]= 1; robot.model.upperPositionLimit = qmax

rmodel = robot.model
rdata = rmodel.createData()


# ---- FLOATING MODEL
# Actuation model is maybe better named transmission model.
# It would be good to write a trivial ActuationModelFull for fully actuated robot, with tau=u.

class ActuationModelFreeFloating:
    '''
    This model transforms an actuation u into a joint torque tau.
    We implement here the simplest model: tau = S.T*u, where S is constant.
    '''

    def __init__(self,pinocchioModel):
        self.pinocchio = pinocchioModel
        if(pinocchioModel.joints[1].shortname() != 'JointModelFreeFlyer'):
            warnings.warn('Strange that the first joint is not a freeflyer')
        self.nq  = pinocchioModel.nq
        self.nv  = pinocchioModel.nv
        self.nx  = self.nq+self.nv
        self.ndx = self.nv*2
        self.nu  = self.nv - 6
    def calc(model,data,x,u):
        data.a[6:] = u
        return data.a
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        return data.a
    def createData(self,pinocchioData):
        return ActuationDataFreeFloating(self,pinocchioData)
        
class ActuationDataFreeFloating:
    def __init__(self,model,pinocchioData):
        self.pinocchio = pinocchioData
        nx,ndx,nq,nv,nu = model.nx,model.ndx,model.nq,model.nv,model.nu
        self.a = np.zeros(nv)                 # result of calc
        self.A = np.zeros([nv,ndx+nu])        # result of calcDiff
        self.Ax = self.A[:,:ndx]
        self.Au = self.A[:,ndx:]
        np.fill_diagonal(self.Au[6:,:],1)
        
actModel = ActuationModelFreeFloating(rmodel)
actData  = actModel.createData(rdata)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv-6))

actModel.calcDiff(actData,x,u)

        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



class DifferentialActionModelActuated:
    '''Unperfect class written to validate the actuation model. Do not use except for tests. '''
    def __init__(self,pinocchioModel,actuationModel):
        self.pinocchio = pinocchioModel
        self.actuation = actuationModel
        self.State = StatePinocchio(self.pinocchio)
        self.costs = CostModelSum(self.pinocchio)
        self.nq,self.nv = self.pinocchio.nq, self.pinocchio.nv
        self.nx = self.State.nx
        self.ndx = self.State.ndx
        self.nout = self.nv
        self.nu = self.actuation.nu
        self.unone = np.zeros(self.nu)
    @property
    def ncost(self): return self.costs.ncost
    def createData(self): return DifferentialActionDataActuated(self)
    def calc(model,data,x,u=None):
        if u is None: u=model.unone
        nx,nu,nq,nv,nout = model.nx,model.nu,model.nq,model.nv,model.nout
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(model.actuation.calc(data.actuation,x,u))
        data.xout[:] = pinocchio.aba(model.pinocchio,data.pinocchio,q,v,tauq).flat
        pinocchio.forwardKinematics(model.pinocchio,data.pinocchio,q,v)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        data.cost = model.costs.calc(data.costs,x,u)
        return data.xout,data.cost

    def calcDiff(model,data,x,u=None,recalc=True):
        if u is None: u=model.unone
        if recalc: xout,cost = model.calc(data,x,u)
        nx,ndx,nu,nq,nv,nout = model.nx,model.State.ndx,model.nu,model.nq,model.nv,model.nout
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(data.actuation.a)
        pinocchio.computeABADerivatives(model.pinocchio,data.pinocchio,q,v,tauq)
        da_dq = data.pinocchio.ddq_dq
        da_dv = data.pinocchio.ddq_dv
        da_dact = data.pinocchio.Minv

        dact_dx = data.actuation.Ax
        dact_du = data.actuation.Au
        
        data.Fx[:,:nv] = da_dq
        data.Fx[:,nv:] = da_dv
        data.Fx += np.dot(da_dact,dact_dx)
        data.Fu[:,:]   = np.dot(da_dact,dact_du)

        pinocchio.computeJointJacobians(model.pinocchio,data.pinocchio,q)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        model.costs.calcDiff(data.costs,x,u,recalc=False)
        
        return data.xout,data.cost

class DifferentialActionDataActuated:
    def __init__(self,model):
        self.pinocchio = model.pinocchio.createData()
        self.actuation = model.actuation.createData(self.pinocchio)
        self.costs = model.costs.createData(self.pinocchio)
        self.cost = np.nan
        self.xout = np.zeros(model.nout)
        nx,nu,ndx,nq,nv,nout = model.nx,model.nu,model.State.ndx,model.nq,model.nv,model.nout
        self.F = np.zeros([ nout,ndx+nu ])
        self.costResiduals = self.costs.residuals
        self.Fx = self.F[:,:ndx]
        self.Fu = self.F[:,-nu:]
        self.g   = self.costs.g
        self.L   = self.costs.L
        self.Lx  = self.costs.Lx
        self.Lu  = self.costs.Lu
        self.Lxx = self.costs.Lxx
        self.Lxu = self.costs.Lxu
        self.Luu = self.costs.Luu
        self.Rx  = self.costs.Rx
        self.Ru  = self.costs.Ru
   
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

assert(absmax(data.Fx-dnum.Fx)/model.nx < 1e-3 )
assert(absmax(data.Fu-dnum.Fu)/model.nu < 1e-3 )



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class ContactModelPinocchio:
    def __init__(self,pinocchioModel,ncontact):
        assert(hasattr(self,'ContactDataType'))
        self.pinocchio = pinocchioModel
        self.nq,self.nv = pinocchioModel.nq,pinocchioModel.nv
        self.nx = self.nq+self.nv
        self.ndx = 2*self.nv
        self.ncontact = ncontact
    def createData(self,pinocchioData):
        return self.ContactDataType(self,pinocchioData)
    def calc(model,data,x):
        assert(False and "This should be defined in the derivative class.")
    def calcDiff(model,data,x,recalc=True):
        assert(False and "This should be defined in the derivative class.")
    def forces(model,data,forcesArr,forcesVec = None):
        '''
        Convert a numpy array of forces into a stdVector of spatial forces.
        If forcesVec is not none, sum the result in it. Otherwise, reset self.fs
        and put the result there.
        '''
        assert(False and "This should be defined in the derivative class.")

class ContactDataPinocchio:
    def __init__(self,model,pinocchioData):
        nc,nq,nv,nx,ndx = model.ncontact,model.nq,model.nv,model.nx,model.ndx
        self.pinocchio = pinocchioData
        self.J = np.zeros([ nc,nv ])
        self.a0 = np.zeros(nc)
        self.Ax = np.zeros([ nc, ndx ])
        self.Aq = self.Ax[:,:nv]
        self.Av = self.Ax[:,nv:]
        self.fs = pinocchio.StdVect_Force()
        for i in range(model.pinocchio.njoints): self.fs.append(pinocchio.Force.Zero())
# -----------------------------------------------------------------------------

class ContactModel6D(ContactModelPinocchio):
    def __init__(self,pinocchioModel,frame,ref):
        self.ContactDataType = ContactData6D
        ContactModelPinocchio.__init__(self,pinocchioModel,ncontact=6)
        self.frame = frame
        self.ref = ref # not used yet ... later
    def calc(model,data,x):
        # We suppose forwardKinematics(q,v,a), computeJointJacobian and updateFramePlacement already
        # computed.
        data.J[:,:] = pinocchio.getFrameJacobian(model.pinocchio,data.pinocchio,
                                                 model.frame,pinocchio.ReferenceFrame.LOCAL)
        data.a0[:] = pinocchio.getFrameAcceleration(model.pinocchio,data.pinocchio,model.frame).vector.flat
    def calcDiff(model,data,x,recalc=True):
        if recalc: model.calc(data,x)
        dv_dq,da_dq,da_dv,da_da = pinocchio.getJointAccelerationDerivatives\
                                  (model.pinocchio,data.pinocchio,data.joint,
                                   pinocchio.ReferenceFrame.LOCAL)
        data.Aq[:,:] = data.fXj*da_dq
        data.Av[:,:] = data.fXj*da_dv
    def forces(model,data,forcesArr,forcesVec=None):
        '''
        Convert a numpy array of forces into a stdVector of spatial forces.
        '''
        # In the dynamic equation, we wrote M*a + J.T*fdyn, while in the ABA it would be
        # M*a + b = tau + J.T faba, so faba = -fdyn (note the minus operator before a2m).
        if forcesVec is None:
            forcesVec = data.fs
            data.fs[data.joint] *= 0
        forcesVec[data.joint] += data.jMf*pinocchio.Force(-a2m(forcesArr))
        return data.fs
    
class ContactData6D(ContactDataPinocchio):
    def __init__(self,model,pinocchioData):
        ContactDataPinocchio.__init__(self,model,pinocchioData)
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent       
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action
        
contactModel = ContactModel6D(rmodel,rmodel.getFrameId('gripper_left_fingertip_2_link'),ref=None)
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
from collections import OrderedDict

# Many contact model
class ContactModelMultiple(ContactModelPinocchio):
    def __init__(self,pinocchioModel):
        self.ContactDataType = ContactDataMultiple
        ContactModelPinocchio.__init__(self,pinocchioModel,ncontact=0)
        self.contacts = OrderedDict()
    def addContact(self,name,contact):
        self.contacts.update([[name,contact]])
        self.ncontact += contact.ncontact
    def calc(model,data,x):
        npast = 0
        for m,d in zip(model.contacts.values(),data.contacts.values()):
            m.calc(d,x)
            data.a0[npast:npast+m.ncontact]   = d.a0
            data.J [npast:npast+m.ncontact,:] = d.J
            npast += m.ncontact
    def calcDiff(model,data,x,recalc=True):
        if recalc: model.calc(data,x)
        npast = 0
        for m,d in zip(model.contacts.values(),data.contacts.values()):
            m.calcDiff(d,x,recalc=False)
            data.Ax[npast:npast+m.ncontact,:]   = d.Ax
            npast += m.ncontact
    def forces(model,data,fsArr):
        npast = 0 
        for i,f in enumerate(data.fs): data.fs[i] *= 0
        for m,d in zip(model.contacts.values(),data.contacts.values()):
            m.forces(d,fsArr[npast:npast+m.ncontact],data.fs)
            npast += m.ncontact
            
class ContactDataMultiple(ContactDataPinocchio):
    def __init__(self,model,pinocchioData):
        ContactDataPinocchio.__init__(self,model,pinocchioData)
        nc,nq,nv,nx,ndx = model.ncontact,model.nq,model.nv,model.nx,model.ndx
        self.contacts = OrderedDict([ [k,m.createData(pinocchioData)] for k,m in model.contacts.items() ])
       
contactModel = ContactModelMultiple(rmodel)
contactModel.addContact('1',ContactModel6D(rmodel,rmodel.getFrameId('gripper_left_fingertip_2_link'),ref=None))

contactData = contactModel.createData(rdata)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)*2-1
x = np.concatenate([ m2a(q),m2a(v) ])
contactModel.calc(contactData,x)
contactModel.calcDiff(contactData,x)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

class DifferentialActionModelFloatingInContact:
    def __init__(self,pinocchioModel,actuationModel,contactModel,costModel):
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.actuation = actuationModel
        self.contact = contactModel
        self.costs = costModel
        self.nq,self.nv = self.pinocchio.nq, self.pinocchio.nv
        self.nx = self.State.nx
        self.ndx = self.State.ndx
        self.nout = self.nv
        self.nu = self.actuation.nu
        self.unone = np.zeros(self.nu)
    @property
    def ncost(self): return self.costs.ncost
    @property
    def ncontact(self): return self.contact.ncontact
    def createData(self): return DifferentialActionDataFloatingInContact(self)
    def calc(model,data,x,u=None):
        if u is None: u=model.unone
        nx,nu,nq,nv,nout,nc = model.nx,model.nu,model.nq,model.nv,model.nout,model.ncontact
        q = a2m(x[:nq])
        v = a2m(x[-nv:])

        pinocchio.computeAllTerms(model.pinocchio,data.pinocchio,q,v)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)

        data.tauq[:] = model.actuation.calc(data.actuation,x,u)
        model.contact.calc(data.contact,x)

        data.K[:nv,:nv] = data.pinocchio.M
        data.K[nv:,:nv] = data.contact.J
        data.K.T[nv:,:nv] = data.contact.J

        data.r[:nv] = data.tauq - m2a(data.pinocchio.nle)
        data.r[nv:] = -data.contact.a0

        data.af[:] = np.dot(inv(data.K),data.r)
        # Convert force array to vector of spatial forces.
        fs = model.contact.forces(data.contact,data.f)

        data.cost = model.costs.calc(data.costs,x,u)
        return data.xout,data.cost

    def calcDiff(model,data,x,u=None,recalc=True):
        if u is None: u=model.unone
        if recalc: xout,cost = model.calc(data,x,u)
        nx,ndx,nu,nq,nv,nout,nc = model.nx,model.State.ndx,model.nu,model.nq,model.nv,model.nout,model.ncontact
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        a = a2m(data.a)
        fs = data.contact.fs

        pinocchio.computeRNEADerivatives(model.pinocchio,data.pinocchio,q,v,a,fs)
        pinocchio.computeForwardKinematicsDerivatives(model.pinocchio,data.pinocchio,q,v,a)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)

        # [a;f] = K^-1 [ tau - b, -gamma ]
        # [a';f'] = -K^-1 [ K'a + b' ; J'a + gamma' ]  = -K^-1 [ rnea'(q,v,a,fs) ; acc'(q,v,a) ]

        # dtau_dq and dtau_dv are the rnea derivatives rnea'
        dtau_dq = data.pinocchio.dtau_dq
        dtau_dv = data.pinocchio.dtau_dv

        # da0_dq and da0_dv are the acceleration derivatives acc'
        model.contact.calcDiff(data.contact,x,recalc=False)
        da0_dq = data.contact.Aq
        da0_dv = data.contact.Av

        # We separate the Kinv into the a and f rows, and the actuation and acceleration columns
        daf_dact = inv(data.K)
        da_dact = daf_dact[:nv,:nv]
        df_dact = daf_dact[nv:,:nv]
        da_da0  = daf_dact[:nv,nv:]
        df_da0  = daf_dact[nv:,nv:]
        
        da_dq = -np.dot(da_dact,dtau_dq) -np.dot(da_da0,da0_dq)
        da_dv = -np.dot(da_dact,dtau_dv) -np.dot(da_da0,da0_dv)

        # tau is a function of x and u (typically trivial in x), whose derivatives are Ax and Au
        dact_dx = data.actuation.Ax
        dact_du = data.actuation.Au
        
        data.Fx[:,:nv] = da_dq
        data.Fx[:,nv:] = da_dv
        data.Fx       += np.dot(da_dact,dact_dx)
        data.Fu[:,:]   = np.dot(da_dact,dact_du)

        model.costs.calcDiff(data.costs,x,u,recalc=False)
        
        return data.xout,data.cost

class DifferentialActionDataFloatingInContact:
    def __init__(self,model):
        self.pinocchio = model.pinocchio.createData()
        self.actuation = model.actuation.createData(self.pinocchio)
        self.contact = model.contact.createData(self.pinocchio)
        self.costs = model.costs.createData(self.pinocchio)
        self.cost = np.nan
        nx,nu,ndx,nq,nv,nout,nc = model.nx,model.nu,model.State.ndx,model.nq,model.nv,model.nout,model.ncontact
        self.F = np.zeros([ nout,ndx+nu ])
        self.costResiduals = self.costs.residuals
        self.Fx = self.F[:,:ndx]
        self.Fu = self.F[:,-nu:]
        self.g   = self.costs.g
        self.L   = self.costs.L
        self.Lx  = self.costs.Lx
        self.Lu  = self.costs.Lu
        self.Lxx = self.costs.Lxx
        self.Lxu = self.costs.Lxu
        self.Luu = self.costs.Luu
        self.Rx  = self.costs.Rx
        self.Ru  = self.costs.Ru

        self.tauq = np.zeros(nv)
        self.K  = np.zeros([nv+nc, nv+nc])  # KKT matrix = [ MJ.T ; J0 ]
        self.r  = np.zeros( nv+nc )         # NLE effects =  [ tau-b ; -gamma ]
        self.af = np.zeros( nv+nc )         # acceleration&forces = [ a ; f ]
        self.a  = self.af[:nv]
        self.f  = self.af[nv:]

        self.xout = self.a

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)*2-1
x = np.concatenate([ m2a(q),m2a(v) ])
u = np.random.rand(rmodel.nv-6)*2-1

actModel = ActuationModelFreeFloating(rmodel)
contactModel = ContactModel6D(rmodel,rmodel.getFrameId('arm_left_7_joint'),ref=None)
contactModel = ContactModel6D(rmodel,rmodel.getFrameId('gripper_left_fingertip_2_link'),ref=None)
rmodel.frames[contactModel.frame].placement = pinocchio.SE3.Random()
contactModel6 = contactModel
contactModel = ContactModelMultiple(rmodel)
contactModel.addContact(name='fingertip',contact=contactModel6)

model = DifferentialActionModelFloatingInContact(rmodel,actModel,contactModel,CostModelSum(rmodel))
data  = model.createData()

model.calc(data,x,u)
assert( len(filter(lambda x:x>0,eig(data.K)[0])) == model.nv )
assert( len(filter(lambda x:x<0,eig(data.K)[0])) == model.ncontact )
_taucheck = pinocchio.rnea(rmodel,rdata,q,v,a2m(data.a),data.contact.fs)
assert( absmax(_taucheck[:6])<1e-6 )
assert( absmax(m2a(_taucheck[6:])-u)<1e-6 )

model.calcDiff(data,x,u)

mnum = DifferentialActionModelNumDiff(model,withGaussApprox=False)
dnum = mnum.createData()
mnum.calcDiff(dnum,x,u)
assert(absmax(data.Fx-dnum.Fx)/model.nx<1e-3)
assert(absmax(data.Fu-dnum.Fu)/model.nu<1e-3)



# --- COMPLETE MODEL WITH COST ----
State = StatePinocchio(rmodel)

actModel = ActuationModelFreeFloating(rmodel)
contactModel = ContactModelMultiple(rmodel)
contactModel.addContact(name='fingertip',contact=contactModel6)

costModel = CostModelSum(rmodel,nu=actModel.nu)
costModel.addCost( name="pos", weight = 10,
                   cost = CostModelPosition(rmodel,nu=actModel.nu,
                                            frame=rmodel.getFrameId('gripper_left_fingertip_2_link'),
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

mnum = refact.ActionModelNumDiff(model,withGaussApprox=True)
dnum = mnum.createData()

model.calc(data,x,u)
model.calcDiff(data,x,u)

mnum.calcDiff(dnum,x,u)
assert( norm(data.Lx-dnum.Lx) < 1e-3 )
assert( norm(data.Lu-dnum.Lu) < 1e-3 )
assert( norm(dnum.Lxx-data.Lxx) < 1e-3)
assert( norm(dnum.Lxu-data.Lxu) < 1e-3)
assert( norm(dnum.Luu-data.Luu) < 1e-3)

