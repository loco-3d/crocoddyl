from differential_action import DifferentialActionDataAbstract, DifferentialActionModelAbstract
from state import StatePinocchio
from utils import a2m, m2a
import numpy as np
import pinocchio



class DifferentialActionModelFloatingInContact(DifferentialActionModelAbstract):
    def __init__(self,pinocchioModel,actuationModel,contactModel,costModel):
        DifferentialActionModelAbstract.__init__(
            self, pinocchioModel.nq, pinocchioModel.nv, actuationModel.nu)
        self.DifferentialActionDataType = DifferentialActionDataFloatingInContact
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.actuation = actuationModel
        self.contact = contactModel
        self.costs = costModel
    @property
    def ncost(self): return self.costs.ncost
    @property
    def ncontact(self): return self.contact.ncontact
    def calc(model,data,x,u=None):
        if u is None: u=model.unone
        nq,nv = model.nq,model.nv
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
        data.Kinv = np.linalg.inv(data.K)

        data.r[:nv] = data.tauq - m2a(data.pinocchio.nle)
        data.r[nv:] = -data.contact.a0
        data.af[:] = np.dot(data.Kinv,data.r)
        data.f[:] *= -1.
        
        # Convert force array to vector of spatial forces.
        fs = model.contact.setForces(data.contact,data.f)

        data.cost = model.costs.calc(data.costs,x,u)
        return data.xout,data.cost

    def calcDiff(model,data,x,u=None,recalc=True):
        if u is None: u=model.unone
        if recalc: xout,cost = model.calc(data,x,u)
        nq,nv = model.nq,model.nv
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        a = a2m(data.a)
        fs = data.contact.forces

        pinocchio.computeRNEADerivatives(model.pinocchio,data.pinocchio,q,v,a,fs)
        pinocchio.computeForwardKinematicsDerivatives(model.pinocchio,data.pinocchio,q,v,a)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)

        # [a;-f] = K^-1 [ tau - b, -gamma ]
        # [a';-f'] = -K^-1 [ K'a + b' ; J'a + gamma' ]  = -K^-1 [ rnea'(q,v,a,fs) ; acc'(q,v,a) ]

        # Derivative of the actuation model tau = tau(q,u)
        # dtau_dq and dtau_dv are the rnea derivatives rnea'
        did_dq = data.pinocchio.dtau_dq
        did_dv = data.pinocchio.dtau_dv

        # Derivative of the contact constraint
        # da0_dq and da0_dv are the acceleration derivatives acc'
        model.contact.calcDiff(data.contact,x,recalc=False)
        dacc_dq = data.contact.Aq
        dacc_dv = data.contact.Av

        data.Kinv = np.linalg.inv(data.K)

        # We separate the Kinv into the a and f rows, and the actuation and acceleration columns
        da_did  = -data.Kinv[:nv,:nv]
        df_did  = data.Kinv[nv:,:nv]
        da_dacc = -data.Kinv[:nv,nv:]
        df_dacc = data.Kinv[nv:,nv:]

        da_dq   =  np.dot(da_did,did_dq) + np.dot(da_dacc,dacc_dq)
        da_dv   =  np.dot(da_did,did_dv) + np.dot(da_dacc,dacc_dv)
        da_dtau =  data.Kinv[:nv,:nv]  # Add this alias just to make the code clearer
        df_dtau =  -data.Kinv[nv:,:nv]  # Add this alias just to make the code clearer

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

    def quasiStatic(self,data,x):
        nu,nq,nv = self.nu,self.nq,self.nv
        if len(x)==nq:
            x = np.concatenate([x,np.zero(nv)])
        else:
            x[nq:] = 0
        self.calcDiff(data,x,np.zeros(nu))
        return np.dot(
            np.linalg.pinv(np.hstack([data.actuation.Au,data.contact.J.T])),
            -data.r[:nv])[:nu]


class DifferentialActionDataFloatingInContact(DifferentialActionDataAbstract):
    def __init__(self,model):
        self.pinocchio = model.pinocchio.createData()
        self.actuation = model.actuation.createData(self.pinocchio)
        self.contact = model.contact.createData(self.pinocchio)
        costData = model.costs.createData(self.pinocchio)
        DifferentialActionDataAbstract.__init__(self, model, costData)

        nu,ndx,nv,nc = model.nu,model.ndx,model.nv,model.ncontact
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
