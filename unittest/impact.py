import pinocchio
from pinocchio.utils import *
import numpy as np
from numpy.linalg import inv,norm,pinv
from continuous import StatePinocchio

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T

# -----------------------------------------------------------------------------
class ImpulseModelPinocchio:
    def __init__(self,pinocchioModel,ncontact):
        assert(hasattr(self,'ImpulseDataType'))
        self.pinocchio = pinocchioModel
        self.nq,self.nv = pinocchioModel.nq,pinocchioModel.nv
        self.nx = self.nq+self.nv
        self.ndx = 2*self.nv
        self.ncontact = ncontact
    def createData(self,pinocchioData):
        return self.ImpulseDataType(self,pinocchioData)
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
        
class ImpulseDataPinocchio:
    def __init__(self,model,pinocchioData):
        nc,nq,nv,nx,ndx = model.ncontact,model.nq,model.nv,model.nx,model.ndx
        self.pinocchio = pinocchioData
        self.J = np.zeros([ nc,nv ])
        self.Jq = np.zeros([ nc,nv ])
        self.f  = np.nan # not set at construction type
        self.forces = pinocchio.StdVect_Force()
        for i in range(model.pinocchio.njoints): self.forces.append(pinocchio.Force.Zero())
        self.Vq = np.zeros([ nc,nv ])
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class ImpulseModel6D(ImpulseModelPinocchio):
    def __init__(self,pinocchioModel,frame):
        self.ImpulseDataType = ImpulseData6D
        ImpulseModelPinocchio.__init__(self,pinocchioModel,ncontact=6)
        self.frame = frame
    def calc(model,data,x):
        # We suppose forwardKinematics(q,v,a), computeJointJacobian and updateFramePlacement already
        # computed.
        data.J[:,:] = pinocchio.getFrameJacobian(model.pinocchio,data.pinocchio,
                                                 model.frame,pinocchio.ReferenceFrame.LOCAL)
    def calcDiff(model,data,x,recalc=True):
        if recalc: model.calc(data,x)
        dv_dq,dv_dv = pinocchio.getJointVelocityDerivatives\
                      (model.pinocchio,data.pinocchio,data.joint,
                       pinocchio.ReferenceFrame.LOCAL)
        data.Vq[:,:] = data.fXj*dv_dq
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
        
class ImpulseData6D(ImpulseDataPinocchio):
    def __init__(self,model,pinocchioData):
        ImpulseDataPinocchio.__init__(self,model,pinocchioData)
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent       
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class ActionModelImpact:
    def __init__(self,pinocchioModel,contactModel):
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.contact = contactModel
        self.nq,self.nv = self.pinocchio.nq, self.pinocchio.nv
        self.nx = self.State.nx
        self.ndx = self.State.ndx
        self.nout = self.nx
        self.nu = 0
        self.unone = np.zeros(self.nu)
    @property
    def ncontact(self): return self.contact.ncontact
    def createData(self): return ActionDataImpact(self)
    def calc(model,data,x,u=None):
        nx,nu,nq,nv,nout,nc = model.nx,model.nu,model.nq,model.nv,model.nout,model.ncontact
        q = a2m(x[:nq])
        v = a2m(x[-nv:])

        pinocchio.computeAllTerms(model.pinocchio,data.pinocchio,q,v)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)

        model.contact.calc(data.contact,x)

        data.K[:nv,:nv] = data.pinocchio.M
        if hasattr(model.pinocchio,'armature'):
            data.K[range(nv),range(nv)] += model.pinocchio.armature.flat
        data.K[nv:,:nv] = data.contact.J
        data.K.T[nv:,:nv] = data.contact.J

        data.r[:nv] = (data.pinocchio.M*v).flat
        data.r[nv:] = 0

        data.af[:] = np.dot(inv(data.K),data.r)
        # Convert force array to vector of spatial forces.
        fs = model.contact.setForces(data.contact,data.f)

        data.xnext[:nq] = q.flat
        data.xnext[nq:] = data.vnext
        
        data.cost = 0
        return data.xnext,data.cost

    def calcDiff(model,data,x,u=None,recalc=True):
        if u is None: u=model.unone
        if recalc: xout,cost = model.calc(data,x,u)
        nx,ndx,nu,nq,nv,nout,nc = model.nx,model.State.ndx,model.nu,model.nq,model.nv,model.nout,model.ncontact
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        vnext = a2m(data.vnext)
        fs = data.contact.forces

        # Derivative M' dv + J'f + b'
        pinocchio.computeRNEADerivatives(model.pinocchio,data.pinocchio,q,v,vnext-v,fs)
        data.did_dq[:,:] = data.pinocchio.dtau_dq
        # Derivative of b'
        pinocchio.computeRNEADerivatives(model.pinocchio,data.pinocchio,q,v,zero(nv))
        data.did_dq[:,:] -= data.pinocchio.dtau_dq

        pinocchio.computeForwardKinematicsDerivatives(model.pinocchio,data.pinocchio,q,v,vnext*0)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)

        # Derivative of the contact constraint
        model.contact.calcDiff(data.contact,x,recalc=False)
        data.dv_dq = data.contact.Vq

        data.Kinv = inv(data.K)

        np.fill_diagonal(data.Fx[:nv,:nv],1)
        data.Fx[:nv,nv:] = 0
        data.Fx[nv:,nv:] = np.dot(data.Kinv[:nv,:nv],data.K[:nv,:nv])
        data.Fx[nv:,:nv] = -np.dot(data.Kinv[:nv,:],np.vstack([data.did_dq, data.dv_dq]))

        return data.xnext,data.cost
    
class ActionDataImpact:
    def __init__(self,model):
        self.pinocchio = model.pinocchio.createData()
        self.contact = model.contact.createData(self.pinocchio)
        self.cost = np.nan
        nx,nu,ndx,nq,nv,nout,nc = model.nx,model.nu,model.State.ndx,model.nq,model.nv,model.nout,model.ncontact
        self.F = np.zeros([ ndx,ndx+nu ])
        self.Fx = self.F[:,:ndx]
        self.Fu = self.F[:,-nu:]
        self.Fq = self.Fx[:,:nv]
        self.Fv = self.Fx[:,nv:]
        self.Fq[:,:] = 0; np.fill_diagonal(self.Fq,1)
        
        self.K  = np.zeros([nv+nc, nv+nc])  # KKT matrix = [ MJ.T ; J0 ]
        self.r  = np.zeros( nv+nc )         # NLE effects =  [ tau-b ; -gamma ]
        self.af = np.zeros( nv+nc )         # acceleration&forces = [ a ; f ]
        self.vnext = self.af[:nv]
        self.f  = self.af[nv:]
        self.did_dq = np.zeros([nv,nv])
        
        self.xnext = np.zeros(nx)



