import rospkg
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

rospack = rospkg.RosPack()
MODEL_PATH = rospack.get_path('talos_data')
#MODEL_PATH = '/home/nmansard/src/cddp/examples'
MESH_DIR = MODEL_PATH
URDF_FILENAME = "talos_left_arm.urdf"
#URDF_MODEL_PATH = MODEL_PATH + "/talos_data/robots/" + URDF_FILENAME
URDF_MODEL_PATH = MODEL_PATH + "/robots/" + URDF_FILENAME

robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(URDF_MODEL_PATH, [MESH_DIR], pinocchio.JointModelFreeFlyer())

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

# --- Fully actuated (trivial) actuation
class ActuationModelFull:
    '''
    This model transforms an actuation u into a joint torque tau.
    We implement here the trivial model: tau = u
    '''

    def __init__(self,pinocchioModel):
        self.pinocchio = pinocchioModel
        self.nq  = pinocchioModel.nq
        self.nv  = pinocchioModel.nv
        self.nx  = self.nq+self.nv
        self.ndx = self.nv*2
        self.nu  = self.nv
    def calc(model,data,x,u):
        data.a[:] = u
        return data.a
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        return data.a
    def createData(self,pinocchioData):
        return ActuationDataFull(self,pinocchioData)
        
class ActuationDataFull:
    def __init__(self,model,pinocchioData):
        self.pinocchio = pinocchioData
        nx,ndx,nq,nv,nu = model.nx,model.ndx,model.nq,model.nv,model.nu
        self.a = np.zeros(nv)                 # result of calc
        self.A = np.zeros([nv,ndx+nu])        # result of calcDiff
        self.Ax = self.A[:,:ndx]
        self.Au = self.A[:,ndx:]
        np.fill_diagonal(self.Au[:,:],1)

actModel = ActuationModelFull(rmodel)
actData  = actModel.createData(rdata)

#u = m2a(rand(rmodel.nv))
#actModel.calcDiff(actData,x,u)

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
    def __init__(self,pinocchioModel,ncontact,nu=None):
        assert(hasattr(self,'ContactDataType'))
        self.pinocchio = pinocchioModel
        self.nq,self.nv = pinocchioModel.nq,pinocchioModel.nv
        self.nx = self.nq+self.nv
        self.ndx = 2*self.nv
        self.nu  = nu
        self.ncontact = ncontact
    def createData(self,pinocchioData):
        return self.ContactDataType(self,pinocchioData)
    def calc(model,data,x):
        assert(False and "This should be defined in the derivative class.")
    def calcDiff(model,data,x,recalc=True):
        assert(False and "This should be defined in the derivative class.")
    def setForces(model,data,forcesArr,forcesVec = None):
        '''
        Convert a numpy array of forces into a stdVector of spatial forces.
        If forcesVec is not none, sum the result in it. Otherwise, reset self.fs
        and put the result there.
        '''
        assert(False and "This should be defined in the derivative class.")
        return self.forces
    def setForcesDiff(model,data,df_dx,df_du):
        '''
        Feed back the derivative model from the action model. 
        '''
        data.df_dx = df_dx
        data.df_du = df_du
        if model.nu is None: model.nu = df_du.shape[1]
        else: assert( df_du.shape[1] == model.nu )
        
class ContactDataPinocchio:
    def __init__(self,model,pinocchioData):
        nc,nq,nv,nx,ndx = model.ncontact,model.nq,model.nv,model.nx,model.ndx
        self.pinocchio = pinocchioData
        self.J = np.zeros([ nc,nv ])
        self.a0 = np.zeros(nc)
        self.Ax = np.zeros([ nc, ndx ])
        self.Aq = self.Ax[:,:nv]
        self.Av = self.Ax[:,nv:]
        self.f  = np.nan # not set at construction type
        self.forces = pinocchio.StdVect_Force()
        for i in range(model.pinocchio.njoints): self.forces.append(pinocchio.Force.Zero())
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
        data.a0[:] = pinocchio.getFrameAcceleration(model.pinocchio,
                                                    data.pinocchio,model.frame).vector.flat
    def calcDiff(model,data,x,recalc=True):
        if recalc: model.calc(data,x)
        dv_dq,da_dq,da_dv,da_da = pinocchio.getJointAccelerationDerivatives\
                                  (model.pinocchio,data.pinocchio,data.joint,
                                   pinocchio.ReferenceFrame.LOCAL)
        data.Aq[:,:] = data.fXj*da_dq
        data.Av[:,:] = data.fXj*da_dv
    def setForces(model,data,forcesArr,forcesVec=None):
        '''
        Convert a numpy array of forces into a stdVector of spatial forces.
        Side effect: keep the force values in data.
        '''
        # In the dynamic equation, we wrote M*a + J.T*fdyn, while in the ABA it would be
        # M*a + b = tau + J.T faba, so faba = -fdyn (note the minus operator before a2m).
        data.f = forcesArr
        if forcesVec is None:
            forcesVec = data.forces
            data.forces[data.joint] *= 0
        forcesVec[data.joint] += data.jMf*pinocchio.Force(-a2m(forcesArr))
        return forcesVec
        
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

class ContactModel3D(ContactModelPinocchio):
    def __init__(self,pinocchioModel,frame,ref):
        self.ContactDataType = ContactData3D
        ContactModelPinocchio.__init__(self,pinocchioModel,ncontact=3)
        self.frame = frame
        self.ref = ref # not used yet ... later
    def calc(model,data,x):
        # We suppose forwardKinematics(q,v,np.zero), computeJointJacobian and updateFramePlacement already
        # computed.
        data.vw[:] = pinocchio.getFrameVelocity(model.pinocchio,
                                             data.pinocchio,model.frame).angular
        data.vv[:] = pinocchio.getFrameVelocity(model.pinocchio,
                                             data.pinocchio,model.frame).linear
        data.J[:,:] = pinocchio.getFrameJacobian(model.pinocchio, data.pinocchio,
                                                 model.frame,
                                                 pinocchio.ReferenceFrame.LOCAL)[:3,:]
        data.Jw[:,:] = pinocchio.getFrameJacobian(model.pinocchio, data.pinocchio,
                                                 model.frame,
                                                 pinocchio.ReferenceFrame.LOCAL)[3:,:]

        data.a0[:] = (pinocchio.getFrameAcceleration(model.pinocchio,
                                                    data.pinocchio,model.frame).linear +\
                                                    cross(data.vw,data.vv)).flat
        
    def calcDiff(model,data,x,recalc=True):
        if recalc: model.calc(data,x)
        dv_dq,da_dq,da_dv,da_da = pinocchio.getJointAccelerationDerivatives\
                                  (model.pinocchio,data.pinocchio,data.joint,
                                   pinocchio.ReferenceFrame.LOCAL)
        data.Aq[:,:] = (data.fXj*da_dq)[:3,:] + \
                       skew(data.vw)*(data.fXj*dv_dq)[:3,:]-\
                       skew(data.vv)*(data.fXj*dv_dq)[3:,:]
        data.Av[:,:] = (data.fXj*da_dv)[:3,:] + skew(data.vw)*data.J-skew(data.vv)*data.Jw
    def setForces(model,data,forcesArr,forcesVec=None):
        '''
        Convert a numpy array of forces into a stdVector of spatial forces.
        '''
        # In the dynamic equation, we wrote M*a + J.T*fdyn, while in the ABA it would be
        # M*a + b = tau + J.T faba, so faba = -fdyn (note the minus operator before a2m).
        data.f = forcesArr
        if forcesVec is None:
            forcesVec = data.forces
            data.forces[data.joint] *= 0
        forcesVec[data.joint] += data.jMf*pinocchio.Force(-a2m(forcesArr), np.zeros((3,1)))
        return forcesVec
    
class ContactData3D(ContactDataPinocchio):
    def __init__(self,model,pinocchioData):
        ContactDataPinocchio.__init__(self,model,pinocchioData)
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent       
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action
        self.vv = np.zeros([ 3,1 ])
        self.vw = np.zeros([ 3,1 ])
        self.Jw = np.zeros([ 3, model.nv ])


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
    def __getitem__(self,key):
        if isinstance(key,str):
            return self.contacts[key]
        elif isinstance(key,ContactModelPinocchio):
            filter = [ v for k,v in self.contacts.items() if v.contact==key ]
            assert(len(filter) == 1 and "The given key is not or not unique in the contact dict. ")
            return filter[0]
        else:
            raise(KeyError("The key should be string or contactmodel."))
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
    def setForces(model,data,fsArr):
        npast = 0 
        for i,f in enumerate(data.forces): data.forces[i] *= 0
        for m,d in zip(model.contacts.values(),data.contacts.values()):
            m.setForces(d,fsArr[npast:npast+m.ncontact],data.forces)
            npast += m.ncontact
        return data.forces
    def setForcesDiff(model,data,df_dx,df_du):
        '''
        Feed back the derivative model from the action model. 
        '''
        if model.nu is None: model.nu = df_du.shape[1]
        else: assert( df_du.shape[1] == model.nu )
        npast = 0 
        for m,d in zip(model.contacts.values(),data.contacts.values()):
            m.setForcesDiff(d,df_dx[npast:npast+m.ncontact,:],df_du[npast:npast+m.ncontact,:])
            npast += m.ncontact
        
class ContactDataMultiple(ContactDataPinocchio):
    def __init__(self,model,pinocchioData):
        ContactDataPinocchio.__init__(self,model,pinocchioData)
        nc,nq,nv,nx,ndx = model.ncontact,model.nq,model.nv,model.nx,model.ndx
        self.model = model
        self.contacts = OrderedDict([ [k,m.createData(pinocchioData)] for k,m in model.contacts.items() ])
    def __getitem__(self,key):
        if isinstance(key,str):
            return self.contacts[key]
        elif isinstance(key,ContactModelPinocchio):
            filter = [ k for k,v in self.model.contacts.items() if v==key ]
            assert(len(filter) == 1 and "The given key is not or not unique in the contact dict. ")
            return self.contacts[filter[0]]
        else:
            raise(KeyError("The key should be string or contactmodel."))
       
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
        if hasattr(model.pinocchio,'armature'):
            data.K[range(nv),range(nv)] += model.pinocchio.armature.flat
        data.K[nv:,:nv] = data.contact.J
        data.K.T[nv:,:nv] = data.contact.J

        data.r[:nv] = data.tauq - m2a(data.pinocchio.nle)
        data.r[nv:] = -data.contact.a0

        data.af[:] = np.dot(inv(data.K),data.r)
        # Convert force array to vector of spatial forces.
        fs = model.contact.setForces(data.contact,data.f)

        data.cost = model.costs.calc(data.costs,x,u)
        return data.xout,data.cost

    def calcDiff(model,data,x,u=None,recalc=True):
        if u is None: u=model.unone
        if recalc: xout,cost = model.calc(data,x,u)
        nx,ndx,nu,nq,nv,nout,nc = model.nx,model.State.ndx,model.nu,model.nq,model.nv,model.nout,model.ncontact
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        a = a2m(data.a)
        fs = data.contact.forces

        pinocchio.computeRNEADerivatives(model.pinocchio,data.pinocchio,q,v,a,fs)
        pinocchio.computeForwardKinematicsDerivatives(model.pinocchio,data.pinocchio,q,v,a)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)

        # [a;f] = K^-1 [ tau - b, -gamma ]
        # [a';f'] = -K^-1 [ K'a + b' ; J'a + gamma' ]  = -K^-1 [ rnea'(q,v,a,fs) ; acc'(q,v,a) ]
        
        # Derivative of the actuation model tau = tau(q,u)
        # dtau_dq and dtau_dv are the rnea derivatives rnea'
        did_dq = data.pinocchio.dtau_dq
        did_dv = data.pinocchio.dtau_dv

        # Derivative of the contact constraint
        # da0_dq and da0_dv are the acceleration derivatives acc'
        model.contact.calcDiff(data.contact,x,recalc=False)
        dacc_dq = data.contact.Aq
        dacc_dv = data.contact.Av

        data.Kinv = inv(data.K)

        # We separate the Kinv into the a and f rows, and the actuation and acceleration columns
        da_did  = -data.Kinv[:nv,:nv]
        df_did  = -data.Kinv[nv:,:nv]
        da_dacc = -data.Kinv[:nv,nv:]
        df_dacc = -data.Kinv[nv:,nv:]
        
        da_dq   =  np.dot(da_did,did_dq) + np.dot(da_dacc,dacc_dq)
        da_dv   =  np.dot(da_did,did_dv) + np.dot(da_dacc,dacc_dv)
        da_dtau =  data.Kinv[:nv,:nv]  # Add this alias just to make the code clearer
        df_dtau =  data.Kinv[nv:,:nv]  # Add this alias just to make the code clearer
        
        # tau is a function of x and u (typically trivial in x), whose derivatives are Ax and Au
        dtau_dx = data.actuation.Ax
        dtau_du = data.actuation.Au
        
        data.Fx[:,:nv] = da_dq
        data.Fx[:,nv:] = da_dv
        data.Fx       += np.dot(da_dtau,dtau_dx)
        data.Fu[:,:]   = np.dot(da_dtau,dtau_du)

        data.df_dq[:,:] = np.dot(df_did,did_dq) + np.dot(df_dacc,dacc_dq)
        data.df_dv[:,:] = np.dot(df_did,did_dv) + np.dot(df_dacc,dacc_dv)
        data.df_dx     += np.dot(df_dtau,dtau_dx)
        data.df_du[:,:] = np.dot(df_dtau,dtau_du)

        model.contact.setForcesDiff(data.contact,data.df_dx,data.df_du)
        
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

        self.df   = np.zeros([nc,ndx+nu])
        self.df_dx = self.df   [:,:ndx]
        self.df_dq = self.df_dx[:,:nv]
        self.df_dv = self.df_dx[:,nv:]
        self.df_du = self.df   [:,ndx:]
        
        self.xout = self.a

#-------------------------------------------------------------

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
contactModel6 = ContactModel6D(rmodel,rmodel.getFrameId('gripper_left_fingertip_2_link'),ref=None)
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
class CostModelForce6D(CostModelPinocchio):
    '''
    The class proposes a model of a cost function for tracking a reference
    value of a 6D force, being given the contact model and its derivatives.
    '''
    def __init__(self,pinocchioModel,contactModel,ref=None,nu=None):
        self.CostDataType = CostDataForce6D
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=6,nu=nu)
        self.ref = ref if ref is not None else np.zeros(6)
        self.contact = contactModel
    def calc(model,data,x,u):
        if data.contact is None:
            raise RunTimeError('''The CostForce data should be specifically initialized from the
            contact data ... no automatic way of doing that yet ...''')
        data.f = data.contact.f
        data.residuals = data.f-model.ref
        data.cost = .5*sum(data.residuals**2)
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        assert(model.nu==len(u) and model.contact.nu == model.nu)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        df_dx,df_du = data.contact.df_dx,data.contact.df_du

        data.Rx [:,:] = df_dx   # This is useless.
        data.Ru [:,:] = df_du   # This is useless

        data.Lx [:]     = np.dot(df_dx.T,data.residuals)
        data.Lu [:]     = np.dot(df_du.T,data.residuals)

        data.Lxx[:,:]   = np.dot(df_dx.T,df_dx)
        data.Lxu[:,:]   = np.dot(df_dx.T,df_du)
        data.Luu[:,:]   = np.dot(df_du.T,df_du)

        return data.cost
    
from continuous import CostDataPinocchio
class CostDataForce6D(CostDataPinocchio):
    def __init__(self,model,pinocchioData,contactData=None):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.contact = contactData

model.costs = CostModelSum(rmodel,nu=actModel.nu)
model.costs.addCost( name='force', weight = 1,
                    cost = CostModelForce6D(rmodel,model.contact.contacts['fingertip'],
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

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# --- COMPLETE MODEL WITH COST ----
State = StatePinocchio(rmodel)

actModel = ActuationModelFreeFloating(rmodel)
contactModel = ContactModelMultiple(rmodel)
contactModel.addContact(name='root_joint',contact=contactModel6)

costModel = CostModelSum(rmodel,nu=actModel.nu)
costModel.addCost( name="pos", weight = 10,
                   cost = CostModelPosition(rmodel,nu=actModel.nu,
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


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# integrative test: checking 1-step DDP versus KKT

from refact import ShootingProblem, SolverDDP,SolverKKT
from continuous import IntegratedActionModelEuler, DifferentialActionModelPositioning


def disp(xs,dt=0.1):
    if not hasattr(robot,'viewer'): robot.initDisplay(loadModel=True)
    import time
    for x in xs:
        robot.display(a2m(x[:robot.nq]))
        time.sleep(dt)

import copy
class SolverLogger:
    def __init__(self):
        self.steps = []
        self.iters = []
        self.costs = []
        self.regularizations = []
        self.xs = []
        self.us = []
    def __call__(self,solver):
        self.xs.append(copy.copy(solver.xs))
        self.steps.append( solver.stepLength )
        self.iters.append( solver.iter )
        self.costs.append( [ d.cost for d in solver.datas() ] )
        self.regularizations.append( solver.x_reg )

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
ddp.callback = SolverLogger()
ddp.th_stop = 1e-18
xddp,uddp,doneddp = ddp.solve(verbose=False,maxiter=30)

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
xkkt,ukkt,donekkt = kkt.solve(init_xs=x0s,init_us=u0s,isFeasible=True,maxiter=20,verbose=False)
xddp,uddp,doneddp = ddp.solve(init_xs=x0s,init_us=u0s,isFeasible=True,maxiter=20,verbose=False)

assert(donekkt)
assert(norm(xkkt[0]-problem.initialState)<1e-9)
assert(norm(xddp[0]-problem.initialState)<1e-9)
for t in range(problem.T):
    assert(norm(ukkt[t]-uddp[t])<1e-6)
    assert(norm(xkkt[t+1]-xddp[t+1])<1e-6)

 
