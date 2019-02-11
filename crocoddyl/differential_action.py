from state import StatePinocchio, StateVector
from utils import a2m, randomOrthonormalMatrix
import numpy as np
from numpy.random import rand
import pinocchio



class DifferentialActionModelAbstract:
    """ Abstract class for the differential action model.

    In crocoddyl, an action model combines dynamics and cost data. Each node, in
    our optimal control problem, is described through an action model. Every
    time that we want describe a problem, we need to provide ways of computing
    the dynamics, cost functions and their derivatives. These computations are
    mainly carry on inside calc() and calcDiff(), respectively.
    """
    def __init__(self, nq, nv, nu):
        self.nq = nq
        self.nv = nv
        self.nu = nu
        self.nx = nq + nv
        self.ndx = 2 * nv
        self.nout = nv
        self.unone = np.zeros(self.nu)

    def createData(self):
        """ Created the differential action data.

        Each differential action model has its own data that needs to be
        allocated. This function returns the allocated data for a predefined
        DAM.
        :return DAM data.
        """
        raise NotImplementedError("Not implemented yet.")

    def calc(model, data, x, u=None):
        """ Compute the state evolution and cost value.

        First, it describes the time-continuous evolution of our dynamical system
        in which along predefined integrated action model we might obtain the
        next discrete state. Indeed it computes the time derivatives of the
        state from a predefined dynamical system. Additionally it computes the
        cost value associated to this state and control pair.
        :param model: differential action model
        :param data: differential action data
        :param x: state vector
        :param u: control input
        """
        raise NotImplementedError("Not implemented yet.")

    def calcDiff(model, data, x, u=None, recalc=True):
        """ Compute the derivatives of the dynamics and cost functions.

        It computes the partial derivatives of the dynamical system and the cost
        function. If recalc == True, it first updates the state evolution and
        cost value. This function builds a quadratic approximation of the
        time-continuous action model (i.e. dynamical system and cost function).
        :param model: differential action model
        :param data: differential action data
        :param x: state vector
        :param u: control input
        :param recalc: If true, it updates the state evolution and the cost
        value.
        """
        raise NotImplementedError("Not implemented yet.")


class DifferentialActionModelFullyActuated(DifferentialActionModelAbstract):
    def __init__(self, pinocchioModel, costModel):
        DifferentialActionModelAbstract.__init__(
            self, pinocchioModel.nq, pinocchioModel.nv, pinocchioModel.nv)
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.costs = costModel
        # Use this to force the computation with ABA
        # Side effect is that armature is not used.
        self.forceAba = False
    @property
    def ncost(self): return self.costs.ncost
    def createData(self): return DifferentialActionDataFullyActuated(self)
    def calc(model,data,x,u=None):
        if u is None: u=model.unone
        nq,nv = model.nq,model.nv
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
        nq,nv = model.nq,model.nv
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

class DifferentialActionDataFullyActuated:
    def __init__(self,model):
        self.pinocchio = model.pinocchio.createData()
        self.costs = model.costs.createData(self.pinocchio)
        self.cost = np.nan
        nu,ndx,nout = model.nu,model.State.ndx,model.nout
        self.xout = np.zeros(nout)
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


class DifferentialActionModelLQR(DifferentialActionModelAbstract):
  """ Differential action model for linear dynamics and quadracti cost.
  
  This class implements a linear dynamics, and quadratic costs (i.e. LQR action).
  Since the DAM is a second order system, and the integratedactionmodels are
  implemented as being second order integrators, This class implements a second
  order linear system given by
    x = [q, v]
    dv = A v + B q + C u + d
  where  A, B, C and d are constant terms.
  
  On the other hand the cost function is given by
    l(x,u) = x^T*Q*x + u^T*U*u
  """
  def __init__(self, nq, nu):
    DifferentialActionModelAbstract.__init__(self, nq, nq, nu)
    self.State = StateVector(self.nx)

    v1 = randomOrthonormalMatrix(self.nq)
    v2 = randomOrthonormalMatrix(self.nq)
    v3 = randomOrthonormalMatrix(self.nq)

    # linear dynamics and quadratic cost terms
    self.A = v2
    self.B = v1
    self.C = v3
    self.d = rand(self.nv)
    self.Q = randomOrthonormalMatrix(self.nx)
    self.U = randomOrthonormalMatrix(self.nu)

  def createData(self):
      return DifferentialActionDataLQR(self)

  def calc(model,data,x,u=None):
    q = x[:model.nq]; v = x[model.nq:]
    data.xout[:] = (np.dot(model.A, v) + np.dot(model.B, q) + np.dot(model.C, u)).flat + model.d
    data.cost = np.dot(x, np.dot(model.Q, x)) + np.dot(u, np.dot(model.U, u))
    return data.xout, data.cost
  
  def calcDiff(model,data,x,u=None,recalc=True):
    if u is None: u=model.unone
    if recalc: xout,cost = model.calc(data,x,u)
    data.Lx[:] = np.dot(x.T, data.Lxx)
    data.Lu[:] = np.dot(u.T, data.Luu)
    return data.xout,data.cost


class DifferentialActionDataLQR:
  def __init__(self,model):
    nx,nu,ndx,nv,nout = model.nx,model.nu,model.State.ndx,model.nv,model.nout
    # State evolution and cost data
    self.cost = np.nan
    self.xout = np.zeros(nout)

    # Dynamics data
    self.F = np.zeros([ nout,ndx+nu ])
    self.Fx = self.F[:,:ndx]
    self.Fu = self.F[:,-nu:]

    # Cost data
    self.g = np.zeros( ndx+nu)
    self.L = np.zeros([ndx+nu,ndx+nu])
    self.Lx = self.g[:ndx]
    self.Lu = self.g[ndx:]
    self.Lxx = self.L[:ndx,:ndx]
    self.Lxu = self.L[:ndx,ndx:]
    self.Luu = self.L[ndx:,ndx:]

    # Setting the linear model and quadratic cost here because they are constant
    self.Fx[:,:model.nv] = model.B
    self.Fx[:,model.nv:] = model.A
    self.Fu[:,:] = model.C
    np.copyto(self.Lxx, model.Q+model.Q.T)
    np.copyto(self.Lxu, np.zeros((nx, nu)))
    np.copyto(self.Luu, model.U+model.U.T)


class DifferentialActionModelNumDiff(DifferentialActionModelAbstract):
    def __init__(self,model,withGaussApprox=False):
        self.model0 = model
        self.nx = model.nx
        self.ndx = model.ndx
        self.nout = model.nout
        self.nu = model.nu
        self.nq = model.nq
        self.nv = model.nv
        self.State = model.State
        self.disturbance = 1e-5
        try:
            self.ncost = model.ncost
        except:
            self.ncost = 1
        self.withGaussApprox = withGaussApprox
        assert( not self.withGaussApprox or self.ncost>1 )

    def createData(self):
        return DifferentialActionDataNumDiff(self)
    def calc(model,data,x,u): return model.model0.calc(data.data0,x,u)
    def calcDiff(model,data,x,u,recalc=True):
        xn0,c0 = model.calc(data,x,u)
        h = model.disturbance
        dist = lambda i,n,h: np.array([ h if ii==i else 0 for ii in range(n) ])
        Xint  = lambda x,dx: model.State.integrate(x,dx)
        for ix in range(model.ndx):
            xn,c = model.model0.calc(data.datax[ix],Xint(x,dist(ix,model.ndx,h)),u)
            data.Fx[:,ix] = (xn-xn0)/h
            data.Lx[  ix] = (c-c0)/h
            if model.ncost>1: data.Rx[:,ix] = (data.datax[ix].costResiduals-data.data0.costResiduals)/h
        if u is not None:
            for iu in range(model.nu):
                xn,c = model.model0.calc(data.datau[iu],x,u+dist(iu,model.nu,h))
                data.Fu[:,iu] = (xn-xn0)/h
                data.Lu[  iu] = (c-c0)/h
                if model.ncost>1: data.Ru[:,iu] = (data.datau[iu].costResiduals-data.data0.costResiduals)/h
        if model.withGaussApprox:
            data.Lxx[:,:] = np.dot(data.Rx.T,data.Rx)
            data.Lxu[:,:] = np.dot(data.Rx.T,data.Ru)
            data.Lux[:,:] = data.Lxu.T
            data.Luu[:,:] = np.dot(data.Ru.T,data.Ru)

class DifferentialActionDataNumDiff:
    def __init__(self,model):
        ndx,nu,nout,ncost = model.ndx,model.nu,model.nout,model.ncost
        self.data0 = model.model0.createData()
        self.datax = [ model.model0.createData() for i in range(model.ndx) ]
        self.datau = [ model.model0.createData() for i in range(model.nu ) ]

        self.g  = np.zeros([ ndx+nu ])
        self.F  = np.zeros([ nout,ndx+nu ])
        self.Lx = self.g[:ndx]
        self.Lu = self.g[ndx:]
        self.Fx = self.F[:,:ndx]
        self.Fu = self.F[:,ndx:]
        if model.ncost >1 :
            self.costResiduals = self.data0.costResiduals
            self.R  = np.zeros([model.ncost,ndx+nu])
            self.Rx = self.R[:,:ndx]
            self.Ru = self.R[:,ndx:]
        if model.withGaussApprox:
            self. L = np.zeros([ ndx+nu, ndx+nu ])
            self.Lxx = self.L[:ndx,:ndx]
            self.Lxu = self.L[:ndx,ndx:]
            self.Lux = self.L[ndx:,:ndx]
            self.Luu = self.L[ndx:,ndx:]