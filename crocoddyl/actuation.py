from state import StatePinocchio
from cost import CostModelSum
from utils import a2m
import numpy as np
import pinocchio
import warnings


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
        self.Lx  = self.costs.Lx
        self.Lu  = self.costs.Lu
        self.Lxx = self.costs.Lxx
        self.Lxu = self.costs.Lxu
        self.Luu = self.costs.Luu
        self.Rx  = self.costs.Rx
        self.Ru  = self.costs.Ru
