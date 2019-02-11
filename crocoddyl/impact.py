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
from crocoddyl.cost import CostModelPinocchio,CostDataPinocchio,CostModelSum
from crocoddyl.activation import ActivationModelQuad


class CostModelImpactBase(CostModelPinocchio):
    def __init__(self,pinocchioModel,ncost):
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=ncost,nu=0)
    def setImpactData(self,data,vnext):
        data.vnext = vnext
    def setImpactDiffData(self,data,dvnext_dx):
        data.dvnext_dx = dvnext_dx
    def assertImpactDataSet(self,data):
        assert(data.vnext is not None \
               and "vnext should be copied first from impact-data. Call setImpactData first")
    def assertImpactDiffDataSet(self,data):
        assert(data.dvnext_dx is not None \
               and "dvnext_dx should be copied first from impact-data. Call setImpactData first")

class CostDataImpactBase(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        # These two fields must be informed by ImpactData.
        self.vnext = None
        self.dvnext_dx = None
    
# --------------------------------------------------------------------------
class CostModelImpactWholeBody(CostModelImpactBase):
    '''
    Penalize the impact on the whole body, i.e. the sum-of-square of ||vnext-v||
    with vnext the velocity after impact and v the velocity before impact.
    '''
    def __init__(self,pinocchioModel,activation=None):
        self.CostDataType = CostDataImpactWholeBody
        CostModelImpactBase.__init__(self,pinocchioModel,ncost=pinocchioModel.nv)
        self.activation = activation if activation is not None else ActivationModelQuad()
    def calc(model,data,x,u=None):
        model.assertImpactDataSet(data)
        nv = model.pinocchio.nv
        data.residuals[:] = data.vnext-x[-nv:]
        data.cost = sum(model.activation.calc(data.activation,data.residuals))
        return data.cost
    def calcDiff(model,data,x,u=None,recalc=True):
        if recalc: model.calc(data,x,u)
        model.assertImpactDiffDataSet(data)
        nv = model.pinocchio.nv
        Ax,Axx = model.activation.calcDiff(data.activation,data.residuals)
        data.Rx[:,:] = data.dvnext_dx
        data.Rx[range(nv),range(nv,2*nv)] -= 1
        data.Lx [:]   = np.dot(data.Rx.T,Ax)
        data.Lxx[:,:] = np.dot(data.Rx.T,Axx*data.Rx)
        
class CostDataImpactWholeBody(CostDataImpactBase):
    def __init__(self,model,pinocchioData):
        CostDataImpactBase.__init__(self,model,pinocchioData)
        self.activation = model.activation.createData()
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0

# --------------------------------------------------------------------------
class CostModelImpactCoM(CostModelImpactBase):
    '''
    Penalize the impact on the com, i.e. the sum-of-square of ||Jcom*(vnext-v)||
    with vnext the velocity after impact and v the velocity before impact.
    '''
    def __init__(self,pinocchioModel,activation=None):
        self.CostDataType = CostDataImpactCoM
        CostModelImpactBase.__init__(self,pinocchioModel,ncost=3)
        self.activation = activation if activation is not None else ActivationModelQuad()
    def calc(model,data,x,u=None):
        model.assertImpactDataSet(data)
        nq,nv = model.pinocchio.nq,model.pinocchio.nv
        pinocchio.centerOfMass(model.pinocchio,data.pinocchio_dv,a2m(x[:nq]),a2m(data.vnext-x[-nv:]))
        data.residuals[:] = data.pinocchio_dv.vcom[0].flat
        data.cost = sum(model.activation.calc(data.activation,data.residuals))
        return data.cost
    def calcDiff(model,data,x,u=None,recalc=True):
        if recalc: model.calc(data,x,u)
        model.assertImpactDiffDataSet(data)
        nq,nv = model.pinocchio.nq,model.pinocchio.nv
        Ax,Axx = model.activation.calcDiff(data.activation,data.residuals)

        ### TODO ???
        # r = Jcom(vnext-v)
        # dr/dv = Jcom*(dvnext/dv - I)
        # dr/dq = dJcom_dq*(vnext-v)   + Jcom*dvnext_dq
        #       = dvcom_dq(vq=vnext-v) + Jcom*dvnext_dq
        # Jcom*v = M[:3,:]/mass * v = RNEA(q,0,v)[:3]/mass
        # => dvcom/dq = dRNEA_dq(q,0,v)[:3,:]/mass
       
        dvc_dq = pinocchio.getCenterOfMassVelocityDerivatives(model.pinocchio,data.pinocchio_dv)
        dvc_dv = pinocchio.jacobianCenterOfMass(model.pinocchio,data.pinocchio_dv)

        # res = vcom(q,vnext-v)
        # dres/dq = dvcom_dq + dvcom_dv*dvnext_dq
        data.Rx[:,:nv] = dvc_dq + np.dot(dvc_dv,data.dvnext_dx[:,:nv])

        # dres/dv = dvcom_dv*(dvnext_dv-I)
        ddv_dv = data.dvnext_dx[:,nv:].copy()
        ddv_dv[range(nv),range(nv)] -= 1
        data.Rx[:,nv:] = np.dot(dvc_dv,ddv_dv)

        data.Lx [:]   = np.dot(data.Rx.T,Ax)
        data.Lxx[:,:] = np.dot(data.Rx.T,Axx*data.Rx)
        
class CostDataImpactCoM(CostDataImpactBase):
    def __init__(self,model,pinocchioData):
        CostDataImpactBase.__init__(self,model,pinocchioData)
        self.activation = model.activation.createData()
        # Those data are ment to be evaluated at v=vnext-v
        self.pinocchio_dv = model.pinocchio.createData()
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

class ActionModelImpact:
    def __init__(self,pinocchioModel,impulseModel,costModel):
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.impulse = impulseModel
        self.nq,self.nv = self.pinocchio.nq, self.pinocchio.nv
        self.nx = self.State.nx
        self.ndx = self.State.ndx
        self.nout = self.nx
        self.nu = 0
        self.unone = np.zeros(self.nu)
        self.costs = costModel
        self.impulseWeight =100.
    @property
    def nimpulse(self): return self.impulse.nimpulse
    @property
    def ncost(self): return self.costs.ncost
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

        if isinstance(model.costs,CostModelImpactBase):
            model.costs.setImpactData(data.costs,data.vnext)
        if isinstance(model.costs,CostModelSum):
            for cmodel,cdata in zip(model.costs.costs.values(),data.costs.costs.values()):
                if isinstance(cmodel.cost,CostModelImpactBase):
                    cmodel.cost.setImpactData(cdata,data.vnext)
            
        data.cost = model.costs.calc(data.costs,x,u=None)
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

        # data.Rx[:,:] = 0
        # np.fill_diagonal(data.Rv,-1)
        # data.Rx[:,:] += data.Fx[nv:,:]
        # data.Rx *= model.impulseWeight
        # data.Lx [:]   = np.dot(data.Rx.T,data.costResiduals)
        # data.Lxx[:,:] = np.dot(data.Rx.T,data.Rx)

        if isinstance(model.costs,CostModelImpactBase):
            model.costs.setImpactDiffData(data.costs,data.Fx[nv:,:])
        if isinstance(model.costs,CostModelSum):
            for cmodel,cdata in zip(model.costs.costs.values(),data.costs.costs.values()):
                if isinstance(cmodel.cost,CostModelImpactBase):
                    cmodel.cost.setImpactDiffData(cdata,data.Fx[nv:,:])
            
        model.costs.calcDiff(data.costs,x,u=None,recalc=recalc)
        
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

        self.costResiduals = self.costs.residuals
        self.g   = self.costs.g
        self.L   = self.costs.L
        self.Lx  = self.costs.Lx
        self.Lu  = self.costs.Lu
        self.Lxx = self.costs.Lxx
        self.Lxu = self.costs.Lxu
        self.Luu = self.costs.Luu
        self.Rx  = self.costs.Rx
        self.Ru  = self.costs.Ru

        self.K  = np.zeros([nv+nc, nv+nc])  # KKT matrix = [ MJ.T ; J0 ]
        self.r  = np.zeros( nv+nc )         # NLE effects =  [ tau-b ; -gamma ]
        self.af = np.zeros( nv+nc )         # acceleration&forces = [ a ; f ]
        self.vnext = self.af[:nv]
        self.f  = self.af[nv:]
        self.did_dq = np.zeros([nv,nv])

        self.xnext = np.zeros(nx)
