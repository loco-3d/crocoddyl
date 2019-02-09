from state import StatePinocchio
from utils import m2a, a2m
import pinocchio
from pinocchio.utils import *
import numpy as np
from numpy.linalg import inv


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class ImpulseModelPinocchio:
    def __init__(self,pinocchioModel,nimpulse):
        assert(hasattr(self,'ImpulseDataType'))
        self.pinocchio = pinocchioModel
        self.nq,self.nv = pinocchioModel.nq,pinocchioModel.nv
        self.nx = self.nq+self.nv
        self.ndx = 2*self.nv
        self.nimpulse = nimpulse
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
        nc,nq,nv,nx,ndx = model.nimpulse,model.nq,model.nv,model.nx,model.ndx
        self.pinocchio = pinocchioData
        self.J = np.zeros([ nc,nv ])
        self.Jq = np.zeros([ nc,nv ])
        self.f  = np.nan # not set at construction type
        self.forces = pinocchio.StdVect_Force()
        for i in range(model.pinocchio.njoints): self.forces.append(pinocchio.Force.Zero())
        self.Vq = np.zeros([ nc,nv ])

# --------------------------------------------------------------------------


class ImpulseModel6D(ImpulseModelPinocchio):
    def __init__(self,pinocchioModel,frame):
        self.ImpulseDataType = ImpulseData6D
        ImpulseModelPinocchio.__init__(self,pinocchioModel,nimpulse=6)
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

# --------------------------------------------------------------------------


class ImpulseModel3D(ImpulseModelPinocchio):
    def __init__(self,pinocchioModel,frame):
        self.ImpulseDataType = ImpulseData3D
        ImpulseModelPinocchio.__init__(self,pinocchioModel,nimpulse=3)
        self.frame = frame
    def calc(model,data,x):
        # We suppose forwardKinematics(q,v,a), computeJointJacobian and updateFramePlacement already
        # computed.
        data.J[:,:] = pinocchio.getFrameJacobian(model.pinocchio,data.pinocchio,
                                                 model.frame,pinocchio.ReferenceFrame.LOCAL)[:3,:]
    def calcDiff(model,data,x,recalc=True):
        if recalc: model.calc(data,x)
        dv_dq,dv_dv = pinocchio.getJointVelocityDerivatives\
                      (model.pinocchio,data.pinocchio,data.joint,
                       pinocchio.ReferenceFrame.LOCAL)
        data.Vq[:,:] = data.fXj[:3,:]*dv_dq
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
        forcesVec[data.joint] += data.jMf*pinocchio.Force(-a2m(forcesArr), np.zeros((3,1)))
        return forcesVec
        
class ImpulseData3D(ImpulseDataPinocchio):
    def __init__(self,model,pinocchioData):
        ImpulseDataPinocchio.__init__(self,model,pinocchioData)
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent       
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action

# --------------------------------------------------------------------------

from collections import OrderedDict
class ImpulseModelMultiple(ImpulseModelPinocchio):
    def __init__(self,pinocchioModel,impulses = {}):
        self.ImpulseDataType = ImpulseDataMultiple
        ImpulseModelPinocchio.__init__(self,pinocchioModel,nimpulse=0)
        self.impulses = OrderedDict()
        for n,i in impulses.items(): self.addImpulse(name=n,impulse=i)
    def addImpulse(self,name,impulse):
        self.impulses.update([[name,impulse]])
        self.nimpulse += impulse.nimpulse
    def __getitem__(self,key):
        if isinstance(key,str):
            return self.impulses[key]
        elif isinstance(key,ImpulseModelPinocchio):
            filter = [ v for k,v in self.impulses.items() if v.impulse==key ]
            assert(len(filter) == 1 and "The given key is not or not unique in the impulse dict. ")
            return filter[0]
        else:
            raise(KeyError("The key should be string or impulsemodel."))
    def calc(model,data,x):
        npast = 0
        for m,d in zip(model.impulses.values(),data.impulses.values()):
            m.calc(d,x)
            data.J [npast:npast+m.nimpulse,:] = d.J
            npast += m.nimpulse
    def calcDiff(model,data,x,recalc=True):
        if recalc: model.calc(data,x)
        npast = 0
        for m,d in zip(model.impulses.values(),data.impulses.values()):
            m.calcDiff(d,x,recalc=False)
            data.Vq[npast:npast+m.nimpulse,:]   = d.Vq
            npast += m.nimpulse
    def setForces(model,data,fsArr):
        npast = 0 
        for i,f in enumerate(data.forces): data.forces[i] *= 0
        for m,d in zip(model.impulses.values(),data.impulses.values()):
            m.setForces(d,fsArr[npast:npast+m.nimpulse],data.forces)
            npast += m.nimpulse
        return data.forces

class ImpulseDataMultiple(ImpulseDataPinocchio):
    def __init__(self,model,pinocchioData):
        ImpulseDataPinocchio.__init__(self,model,pinocchioData)
        nc,nq,nv,nx,ndx = model.nimpulse,model.nq,model.nv,model.nx,model.ndx
        self.model = model
        self.impulses = OrderedDict([ [k,m.createData(pinocchioData)] for k,m in model.impulses.items() ])
    def __getitem__(self,key):
        if isinstance(key,str):
            return self.impulses[key]
        elif isinstance(key,ImpulseModelPinocchio):
            filter = [ k for k,v in self.model.impulses.items() if v==key ]
            assert(len(filter) == 1 and "The given key is not or not unique in the impulse dict. ")
            return self.impulses[filter[0]]
        else:
            raise(KeyError("The key should be string or impulsemodel."))

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class ActionModelImpact:
    def __init__(self,pinocchioModel,impulseModel,costModel=None):
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.impulse = impulseModel
        self.nq,self.nv = self.pinocchio.nq, self.pinocchio.nv
        self.nx = self.State.nx
        self.ndx = self.State.ndx
        self.nout = self.nx
        self.nu = 0
        self.unone = np.zeros(self.nu)
        self.ncost = self.nv
        self.costs = costModel
        self.impulseWeight =100.
    @property
    def nimpulse(self): return self.impulse.nimpulse
    def createData(self): return ActionDataImpact(self)
    def calc(model,data,x,u=None):
        '''
        M(vnext-v) + J^T f = 0
        J vnext = 0

        [MJ^T][vnext] = [Mv]
        [J   ][ f   ]   [0 ]

        [vnext] = K^-1[Mv], with K = [MJ^T;J0]
        [ f   ]       [0 ]
        '''
        nx,nu,nq,nv,nout,nc = model.nx,model.nu,model.nq,model.nv,model.nout,model.nimpulse
        q = a2m(x[:nq])
        v = a2m(x[-nv:])

        pinocchio.computeAllTerms(model.pinocchio,data.pinocchio,q,v)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)

        model.impulse.calc(data.impulse,x)

        data.K[:nv,:nv] = data.pinocchio.M
        if hasattr(model.pinocchio,'armature'):
            data.K[range(nv),range(nv)] += model.pinocchio.armature.flat
        data.K[nv:,:nv] = data.impulse.J
        data.K.T[nv:,:nv] = data.impulse.J

        data.r[:nv] = (data.K[:nv,:nv]*v).flat
        data.r[nv:] = 0
 
        data.af[:] = np.dot(inv(data.K),data.r)
        # Convert force array to vector of spatial forces.
        fs = model.impulse.setForces(data.impulse,data.f)

        data.xnext[:nq] = q.flat
        data.xnext[nq:] = data.vnext

        data.costResiduals[:] = model.impulseWeight*(data.vnext-v.flat)
        data.cost = .5*sum( data.costResiduals**2 )

        if model.costs is not None:
            data.cost += model.costs.calc(data.costs,x,u=None)
        return data.xnext,data.cost

    def calcDiff(model,data,x,u=None,recalc=True):
        '''
        k = [Mv;0]; K = [MJ^T;J0]
        r = [vnext;f] = K^-1 k
        dr/dv = K^-1 [M;0]
        dr/dq = -K^-1 K'K^-1 k + K^-1 k' = -K^-1 (K'r-k')
              = -K^-1 [ M'vnext + J'^T f- M'v ]
                      [ J'vnext               ]
              = -K^-1 [ M'(vnext-v) + J'^T f ]
                      [ J' vnext             ]
        '''
        if recalc: xout,cost = model.calc(data,x,u)
        nx,ndx,nq,nv,nc = model.nx,model.State.ndx,model.nq,model.nv,model.nimpulse
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        vnext = a2m(data.vnext)
        fs = data.impulse.forces

        # Derivative M' dv + J'f + b'
        g6bak = model.pinocchio.gravity.copy()
        model.pinocchio.gravity = pinocchio.Motion.Zero()
        pinocchio.computeRNEADerivatives(model.pinocchio,data.pinocchio,q,zero(nv),vnext-v,fs)
        model.pinocchio.gravity = g6bak
        data.did_dq[:,:] = data.pinocchio.dtau_dq

        # Derivative of the impulse constraint
        pinocchio.computeForwardKinematicsDerivatives(model.pinocchio,data.pinocchio,q,vnext,zero(nv))
        #pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        model.impulse.calcDiff(data.impulse,x,recalc=False)
        data.dv_dq = data.impulse.Vq

        data.Kinv = inv(data.K)

        data.Fq[:nv,:] = 0
        np.fill_diagonal(data.Fq[:nv,:],1)  # dq/dq
        data.Fv[:nv,:] = 0                  # dq/dv
        data.Fx[nv:,:nv] = -np.dot(data.Kinv[:nv,:],np.vstack([data.did_dq, data.dv_dq]))  # dvnext/dq
        data.Fx[nv:,nv:] = np.dot(data.Kinv[:nv,:nv],data.K[:nv,:nv])                      # dvnext/dv

        data.Rx[:,:] = 0
        np.fill_diagonal(data.Rv,-1)
        data.Rx[:,:] += data.Fx[nv:,:]
        data.Rx *= model.impulseWeight
        data.Lx [:]   = np.dot(data.Rx.T,data.costResiduals)
        data.Lxx[:,:] = np.dot(data.Rx.T,data.Rx)

        if model.costs is not None:
            model.costs.calcDiff(data.costs,x,u=None)
            data.Lx[:] += data.costs.Lx
            data.Lxx[:,:] += data.costs.Lxx
            
        return data.xnext,data.cost
    
class ActionDataImpact:
    def __init__(self,model):
        self.pinocchio = model.pinocchio.createData()
        self.impulse = model.impulse.createData(self.pinocchio)
        if model.costs is not None:
            self.costs = model.costs.createData(self.pinocchio)
        self.cost = np.nan
        nx,nu,ndx,nq,nv,nout,nc = model.nx,model.nu,model.State.ndx,model.nq,model.nv,model.nout,model.nimpulse
        self.F = np.zeros([ ndx,ndx+nu ])
        self.Fx = self.F[:,:ndx]
        self.Fu = self.F[:,ndx:]
        self.Fq = self.Fx[:,:nv]
        self.Fv = self.Fx[:,nv:]
        self.Fq[:,:] = 0; np.fill_diagonal(self.Fq,1)

        self.costResiduals = np.zeros(nv)
        self.Rx = np.zeros([ nv,ndx ])
        self.Rq = self.Rx[:,:nv]
        self.Rv = self.Rx[:,nv:]

        self.g = np.zeros(ndx+nu)
        self.L = np.zeros([ndx+nu,ndx+nu])
        self.Lx  = self.g[:ndx]
        self.Lu  = self.g[ndx:]
        self.Lxx = self.L[:ndx,:ndx]
        self.Lxu = self.L[:ndx,ndx:]
        self.Luu = self.L[ndx:,ndx:]

        self.K  = np.zeros([nv+nc, nv+nc])  # KKT matrix = [ MJ.T ; J0 ]
        self.r  = np.zeros( nv+nc )         # NLE effects =  [ tau-b ; -gamma ]
        self.af = np.zeros( nv+nc )         # acceleration&forces = [ a ; f ]
        self.vnext = self.af[:nv]
        self.f  = self.af[nv:]
        self.did_dq = np.zeros([nv,nv])

        self.xnext = np.zeros(nx)
