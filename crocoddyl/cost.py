from activation import ActivationModelQuad
from utils import m2a
import numpy as np
import pinocchio
from collections import OrderedDict



class CostModelPinocchio:
    '''
    This class defines a template of cost model whose function and derivatives
    can be evaluated from pinocchio data only (no need to recompute anything
    in particular to be given the variables x,u).
    '''
    def __init__(self,pinocchioModel,ncost,withResiduals=True,nu=None):
        self.ncost = ncost
        self.nq  = pinocchioModel.nq
        self.nv  = pinocchioModel.nv
        self.nx  = self.nq+self.nv
        self.ndx = self.nv+self.nv
        self.nu  = nu if nu is not None else pinocchioModel.nv
        self.pinocchio = pinocchioModel
        self.withResiduals=withResiduals

    def createData(self,pinocchioData):
        return self.CostDataType(self,pinocchioData)
    def calc(model,data,x,u):
        assert(False and "This should be defined in the derivative class.")
    def calcDiff(model,data,x,u,recalc=True):
        assert(False and "This should be defined in the derivative class.")

class CostDataPinocchio:
    '''
    Abstract data class corresponding to the abstract model class
    CostModelPinocchio.
    '''
    def __init__(self,model,pinocchioData):
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        self.pinocchio = pinocchioData
        self.cost = np.nan
        self.g = np.zeros( ndx+nu)
        self.L = np.zeros([ndx+nu,ndx+nu])

        self.Lx = self.g[:ndx]
        self.Lu = self.g[ndx:]
        
        self.Lxx = self.L[:ndx,:ndx]
        self.Lxu = self.L[:ndx,ndx:]
        self.Luu = self.L[ndx:,ndx:]

        self.Lq  = self.Lx [:nv]
        self.Lqq = self.Lxx[:nv,:nv]
        self.Lv  = self.Lx [nv:]
        self.Lvv = self.Lxx[nv:,nv:]

        if model.withResiduals:
            self.residuals = np.zeros(ncost)
            self.R  = np.zeros([ncost,ndx+nu])
            self.Rx = self.R[:,:ndx]
            self.Ru = self.R[:,ndx:]
            self.Rq  = self.Rx [:,  :nv]
            self.Rv  = self.Rx [:,  nv:]



class CostModelNumDiff(CostModelPinocchio):
    def __init__(self,costModel,State,withGaussApprox=False,reevals=[]):
        '''
        reevals is a list of lambdas of (pinocchiomodel,pinocchiodata,x,u) to be
        reevaluated at each num diff.
        '''
        self.CostDataType = CostDataNumDiff
        CostModelPinocchio.__init__(self,costModel.pinocchio,ncost=costModel.ncost,nu=costModel.nu)
        self.State = State
        self.model0 = costModel
        self.disturbance = 1e-6
        self.withGaussApprox = withGaussApprox
        if withGaussApprox: assert(costModel.withResiduals)
        self.reevals = reevals
    def calc(model,data,x,u):
        data.cost = model.model0.calc(data.data0,x,u)
        if model.withGaussApprox: data.residuals = data.data0.residuals
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        h = model.disturbance
        dist = lambda i,n,h: np.array([ h if ii==i else 0 for ii in range(n) ])
        Xint  = lambda x,dx: model.State.integrate(x,dx)
        for ix in range(ndx):
            xi = Xint(x,dist(ix,ndx,h))
            [ r(model.model0.pinocchio,data.datax[ix].pinocchio,xi,u) for r in model.reevals ]
            c = model.model0.calc(data.datax[ix],xi,u)
            data.Lx[ix] = (c-data.data0.cost)/h
            if model.withGaussApprox:
                data.Rx[:,ix] = (data.datax[ix].residuals-data.data0.residuals)/h
        for iu in range(nu):
            ui = u + dist(iu,nu,h)
            [ r(model.model0.pinocchio,data.datau[iu].pinocchio,x,ui) for r in model.reevals ]
            c = model.model0.calc(data.datau[iu],x,ui)
            data.Lu[iu] = (c-data.data0.cost)/h
            if model.withGaussApprox:
                data.Ru[:,iu] = (data.datau[iu].residuals-data.data0.residuals)/h
        if model.withGaussApprox:
            data.L[:,:] = np.dot(data.R.T,data.R)

class CostDataNumDiff(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        self.pinocchio = pinocchioData
        self.data0 = model.model0.createData(pinocchioData)
        self.datax = [ model.model0.createData(model.model0.pinocchio.createData()) for i in range(nx) ]
        self.datau = [ model.model0.createData(model.model0.pinocchio.createData()) for i in range(nu) ]



class CostModelSum(CostModelPinocchio):
    # This could be done with a namedtuple but I don't like the read-only labels.
    class CostItem:
        def __init__(self,name,cost,weight):
            self.name = name; self.cost = cost; self.weight = weight
        def __str__(self):
            return "CostItem(name=%s, cost=%s, weight=%s)" \
                % ( str(self.name),str(self.cost.__class__),str(self.weight) )
        __repr__=__str__
    def __init__(self,pinocchioModel,nu=None,withResiduals=True):
        self.CostDataType = CostDataSum
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=0,nu=nu)
        # Preserve task order in evaluation, which is a nicer behavior when debuging.
        self.costs = OrderedDict()
    def addCost(self,name,cost,weight):
        assert( cost.withResiduals and \
                '''The cost-of-sums class has not been designed nor tested for non sum of squares
                cost functions. It should not be a big deal to modify it, but this is not done
                yet. ''' )
        self.costs.update([[name,self.CostItem(cost=cost,name=name,weight=weight)]])
        self.ncost += cost.ncost
    def __getitem__(self,key):
        if isinstance(key,str):
            return self.costs[key]
        elif isinstance(key,CostModelPinocchio):
            filter = [ v for k,v in self.costs.items() if v.cost==key ]
            assert(len(filter) == 1 and "The given key is not or not unique in the costs dict. ")
            return filter[0]
        else:
            raise(KeyError("The key should be string or costmodel."))
    def calc(model,data,x,u):
        data.cost = 0
        nr = 0
        for m,d in zip(model.costs.values(),data.costs.values()):
            data.cost += m.weight*m.cost.calc(d,x,u)
            if model.withResiduals:
                data.residuals[nr:nr+m.cost.ncost] = np.sqrt(m.weight)*d.residuals
                nr += m.cost.ncost
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        data.g.fill(0)
        data.L.fill(0)
        nr = 0
        for m,d in zip(model.costs.values(),data.costs.values()):
            m.cost.calcDiff(d,x,u,recalc=False)
            data.Lx [:] += m.weight*d.Lx
            data.Lu [:] += m.weight*d.Lu
            data.Lxx[:] += m.weight*d.Lxx
            data.Lxu[:] += m.weight*d.Lxu
            data.Luu[:] += m.weight*d.Luu
            if model.withResiduals:
                data.Rx[nr:nr+m.cost.ncost] = np.sqrt(m.weight)*d.Rx
                data.Ru[nr:nr+m.cost.ncost] = np.sqrt(m.weight)*d.Ru
                nr += m.cost.ncost
        return data.cost

class CostDataSum(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.model = model
        self.costs = OrderedDict([ [i.name, i.cost.createData(pinocchioData)] \
                                   for i in model.costs.values() ])
    def __getitem__(self,key):
        if isinstance(key,str):
            return self.costs[key]
        elif isinstance(key,CostModelPinocchio):
            filter = [ k for k,v in self.model.costs.items() if v.cost==key ]
            assert(len(filter) == 1 and "The given key is not or not unique in the costs dict. ")
            return self.costs[filter[0]]
        else:
            raise(KeyError("The key should be string or costmodel."))



class CostModelPosition(CostModelPinocchio):
    '''
    The class proposes a model of a cost function positioning (3d) 
    a frame of the robot. Paramterize it with the frame index frameIdx and
    the effector desired position ref.
    '''
    def __init__(self,pinocchioModel,frame,ref,nu=None,activation=None):
        self.CostDataType = CostDataPosition
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=3,nu=nu)
        self.ref = ref
        self.frame = frame
        self.activation = activation if activation is not None else ActivationModelQuad()
    def calc(model,data,x,u):
        data.residuals = m2a(data.pinocchio.oMf[model.frame].translation) - model.ref
        data.cost = sum(model.activation.calc(data.activation,data.residuals))
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        R = data.pinocchio.oMf[model.frame].rotation
        J = R*pinocchio.getFrameJacobian(model.pinocchio,data.pinocchio,model.frame,
                                         pinocchio.ReferenceFrame.LOCAL)[:3,:]
        Ax,Axx = model.activation.calcDiff(data.activation,data.residuals)
        data.Rq[:,:nq] = J
        data.Lq[:]     = np.dot(J.T,Ax)
        data.Lqq[:,:]  = np.dot(data.Rq.T,Axx*data.Rq) # J is a matrix, use Rq instead.
        return data.cost

class CostDataPosition(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.activation = model.activation.createData()
        self.Lu = 0
        self.Lv = 0
        self.Lxu = 0
        self.Luu = 0
        self.Lvv = 0
        self.Ru = 0
        self.Rv = 0



class CostModelFrameVelocity(CostModelPinocchio):
    '''
    The class proposes a model of a cost function that penalize the velocity of a given 
    effector.
    Assumes updateFramePlacement and computeForwardKinematicsDerivatives.
    '''
    def __init__(self,pinocchioModel,frame,ref = None):
        self.CostDataType = CostDataFrameVelocity
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=6)
        self.ref = ref if ref is not None else np.zeros(6) 
        self.frame = frame
    def calc(model,data,x,u):
        data.residuals[:] = m2a(pinocchio.getFrameVelocity(model.pinocchio,data.pinocchio,
                                                           model.frame).vector) - model.ref
        data.cost = .5*sum(data.residuals**2)
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        dv_dq,dv_dvq = pinocchio.getJointVelocityDerivatives\
                                  (model.pinocchio,data.pinocchio,data.joint,
                                   pinocchio.ReferenceFrame.LOCAL)
        data.Rq[:,:] = data.fXj*dv_dq
        data.Rv[:,:] = data.fXj*dv_dvq
        data.Lx[:]     = np.dot(data.Rx.T,data.residuals)
        data.Lxx[:,:]  = np.dot(data.Rx.T,data.Rx)
        return data.cost

class CostDataFrameVelocity(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent       
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0



class CostModelFramePlacement(CostModelPinocchio):
    '''
   The class proposes a model of a cost function position and orientation (6d) 
    for a frame of the robot. Paramterize it with the frame index frameIdx and
    the effector desired pinocchio::SE3 ref.
    '''
    def __init__(self,pinocchioModel,frame,ref,nu=None):
        self.CostDataType = CostDataFramePlacement
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=6,nu=nu)
        self.ref = ref
        self.frame = frame
    def calc(model,data,x,u):
        data.rMf = model.ref.inverse()*data.pinocchio.oMf[model.frame]
        data.residuals[:] = m2a(pinocchio.log(data.rMf).vector)
        data.cost = .5*sum(data.residuals**2)
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        J = np.dot(pinocchio.Jlog6(data.rMf),
                      pinocchio.getFrameJacobian(model.pinocchio,
                                                 data.pinocchio,
                                                 model.frame,
                                                 pinocchio.ReferenceFrame.LOCAL))
        data.Rq[:,:nq] = J
        data.Lq[:]     = np.dot(J.T,data.residuals)
        data.Lqq[:,:]  = np.dot(J.T,J)
        return data.cost

class CostDataFramePlacement(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.rMf = None
        self.Lu = 0
        self.Lv = 0
        self.Lxu = 0
        self.Luu = 0
        self.Lvv = 0
        self.Ru = 0
        self.Rv = 0



class CostModelCoM(CostModelPinocchio):
    '''
    The class proposes a model of a cost function CoM.
    Paramterize it with the desired CoM ref
    '''
    def __init__(self,pinocchioModel,ref,nu=None):
        self.CostDataType = CostDataCoM
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=3,nu=nu)
        self.ref = ref
    def calc(model,data,x,u):
        data.residuals = m2a(data.pinocchio.com[0]) - model.ref
        data.cost = .5*sum(data.residuals**2)
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu

        J = data.pinocchio.Jcom
        data.Rq[:,:nq] = J
        data.Lq[:]     = np.dot(J.T,data.residuals)
        data.Lqq[:,:]  = np.dot(J.T,J)
        return data.cost

class CostDataCoM(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.Lu = 0
        self.Lv = 0
        self.Lxu = 0
        self.Luu = 0
        self.Lvv = 0
        self.Ru = 0
        self.Rv = 0



class CostModelState(CostModelPinocchio):
    def __init__(self,pinocchioModel,State,ref,nu=None):
        self.CostDataType = CostDataState
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=State.ndx,nu=nu)
        self.State = State
        self.ref = ref
        self.weights = None
    def calc(model,data,x,u):
        w = (1 if model.weights is None else model.weights)
        data.residuals[:] = w*model.State.diff(model.ref,x)
        data.cost = .5*sum(data.residuals**2)
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        w = (1 if model.weights is None else model.weights)
        data.Rx[:,:] = (w*model.State.Jdiff(model.ref,x,'second').T).T
        data.Lx[:] = np.dot(data.Rx.T,data.residuals)
        data.Lxx[:,:] = np.dot(data.Rx.T,data.Rx)

class CostDataState(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0



class CostModelControl(CostModelPinocchio):
    def __init__(self,pinocchioModel,nu=None,ref=None):
        self.CostDataType = CostDataControl
        nu = nu if nu is not None else pinocchioModel.nv
        if ref is not None: assert( ref.shape == (nu,) )
        CostModelPinocchio.__init__(self,pinocchioModel,nu=nu,ncost=nu)
        self.ref = ref
    def calc(model,data,x,u):
        data.residuals[:] = u if model.ref is None else u-model.ref
        data.cost = .5*sum(data.residuals**2)
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        #data.Ru[:,:] = np.eye(nu)
        data.Lu[:] = data.residuals
        #data.Luu[:,:] = data.Ru
        assert( data.Luu[0,0] == 1 and data.Luu[1,0] == 0 )

class CostDataControl(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        self.Lx = 0
        self.Lxx = 0
        self.Lxu = 0
        self.Rx = 0
        self.Luu[:,:] = np.eye(nu)
        self.Ru [:,:] = self.Luu

class CostModelSoftStateLimits(CostModelPinocchio):
    '''
    The class assumes:
    1) NFreeflyer joint
    2) Only rotational joints

    These assumptions are hardcoded in the implementation.

    The class proposes a soft quadratic barrier on joint limits and velocity limits
    The joint upper and lower limits, and the joint velocity limits are taken from
    pinocchioModel.(upper)(lower)PositionLimit and pinocchioModel.velocityLimit

    x_mrange = (upper-lower)/2
    xm = lower+ x_range
    x0 = xm - beta*x_mrange
    x1 = xm + beta*x_mrange

    data.cost = 0   x0<=x<=x1
    data.cost = 1.*(x0-x)**2/(x0-lower)**2      x <= x0
    data.cost = 1.*(x-x1)**2/(upper-x1)**2      x >= x1
    '''

    def __init__(self,pinocchioModel,beta=0.9, nu=None):
        self.CostDataType = CostDataSoftStateLimits
        CostModelPinocchio.__init__(self,pinocchioModel,
                                    ncost=pinocchioModel.nq+pinocchioModel.nv,nu=nu)
        assert(beta<1.0)
        self.beta = beta
        self.lowerLimit = np.array(np.vstack([pinocchioModel.lowerPositionLimit,
                                              -pinocchioModel.velocityLimit])).squeeze()
        self.upperLimit = np.array(np.vstack([pinocchioModel.upperPositionLimit,
                                              pinocchioModel.velocityLimit])).squeeze()

        self.x_mrange = (self.upperLimit-self.lowerLimit)/2
        self.xm = self.lowerLimit + self.x_mrange
        self.x0 = self.xm - self.beta*self.x_mrange
        self.x1 = self.xm + self.beta*self.x_mrange
        self.weights = None
        
    def calc(model,data,x,u):
        w = (1 if model.weights is None else model.weights)
        data.lower_residuals = \
                         np.maximum(model.x0-x, 0.)/(model.x0-model.lowerLimit)
        #data.lower_residuals[range(7)+range(model.nq+6)] = 0.

        data.upper_residuals = \
                         np.maximum(x-model.x1, 0.)/(model.upperLimit-model.x1)
        
        #data.upper_residuals[range(7)+range(model.nq+6)] = 0.

        data.residuals[:] = w*(data.lower_residuals+data.upper_residuals)

        data.cost = .5*sum(data.residuals**2)
        return data.cost

    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        w = (1 if model.weights is None else model.weights)

        data.Rx_lower = -1./(model.x0-model.lowerLimit)
        data.Rx_lower[(model.x0-x)<0.] = 0.# ; data.Rx_lower[range(6)+range(model.nv+6)] = 0.

        data.Rx_upper = 1./(model.upperLimit-model.x1)
        data.Rx_upper[(x-model.x1)<0.] = 0.#;  data.Rx_upper[range(6)+range(model.nv+6)] = 0.
        
        data.Rx[:,:] = w*np.diag(data.Rx_upper + data.Rx_lower)

        data.Lx[:] = np.dot(data.Rx.T,data.residuals)
        data.Lxx[:,:] = np.dot(data.Rx.T,data.Rx)

class CostDataSoftStateLimits(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0



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

class CostDataForce6D(CostDataPinocchio):
    def __init__(self,model,pinocchioData,contactData=None):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.contact = contactData
