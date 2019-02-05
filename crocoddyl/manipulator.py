from state import StatePinocchio
from cost import CostModelSum
from utils import a2m
import numpy as np
import pinocchio


class DifferentialActionModelManipulator:
    def __init__(self,pinocchioModel):
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.costs = CostModelSum(self.pinocchio)
        self.nq,self.nv = self.pinocchio.nq, self.pinocchio.nv
        self.nx = self.State.nx
        self.ndx = self.State.ndx
        self.nout = self.nv
        self.nu = self.nv
        self.unone = np.zeros(self.nu)
        # Use this to force the computation with ABA
        # Side effect is that armature is not used.
        self.forceAba = False
    @property
    def ncost(self): return self.costs.ncost
    def createData(self): return DifferentialActionDataManipulator(self)
    def calc(model,data,x,u=None):
        if u is None: u=model.unone
        nx,nu,nq,nv,nout = model.nx,model.nu,model.nq,model.nv,model.nout
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(u)
        # --- Dynamics
        if model.forceAba:
            data.xout[:] = pinocchio.aba(model.pinocchio,data.pinocchio,q,v,tauq).flat
        else:
            pinocchio.computeAllTerms(model.pinocchio,data.pinocchio,q,v)
            data.M = data.pinocchio.M
            if hasattr(model.pinocchio,'armature'):
                data.M[range(nv),range(nv)] += model.pinocchio.armature.flat
            data.Minv = np.linalg.inv(data.M)
            data.xout[:] = data.Minv*(tauq-data.pinocchio.nle).flat
        # --- Cost
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
        tauq = a2m(u)
        a = a2m(data.xout)
        # --- Dynamics
        if model.forceAba:
            pinocchio.computeABADerivatives(model.pinocchio,data.pinocchio,q,v,tauq)
            data.Fx[:,:nv] = data.pinocchio.ddq_dq
            data.Fx[:,nv:] = data.pinocchio.ddq_dv
            data.Fu[:,:]   = pinocchio.computeMinverse(model.pinocchio,data.pinocchio,q)
        else:
            pinocchio.computeRNEADerivatives(model.pinocchio,data.pinocchio,q,v,a)
            data.Fx[:,:nv] = -np.dot(data.Minv,data.pinocchio.dtau_dq)
            data.Fx[:,nv:] = -np.dot(data.Minv,data.pinocchio.dtau_dv)
            data.Fu[:,:] = data.Minv
        # --- Cost
        pinocchio.computeJointJacobians(model.pinocchio,data.pinocchio,q)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        model.costs.calcDiff(data.costs,x,u,recalc=False)

        return data.xout,data.cost

class DifferentialActionDataManipulator:
    def __init__(self,model):
        self.pinocchio = model.pinocchio.createData()
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
