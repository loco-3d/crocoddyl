from state import StateVector
import numpy as np



class ActionDataLQR:
    def __init__(self,actionModel):
        assert( isinstance(actionModel,ActionModelLQR) )
        self.model = actionModel

        self.xnext = np.zeros([ self.model.nx ])
        self.cost  = np.nan
        self.Lx  = np.zeros([ self.model.ndx ])
        self.Lu  = np.zeros([ self.model.nu  ])
        self.Lxx = self.model.Lxx
        self.Lxu = self.model.Lxu
        self.Luu = self.model.Luu
        self.Fx  = self.model.Fx
        self.Fu  = self.model.Fu

class ActionModelLQR:
    def __init__(self,nx,nu):
        '''
        Transition model is xnext(x,u) = Fx*x + Fu*x.
        Cost model is cost(x,u) = 1/2 [x,u].T [Lxx Lxu ; Lxu.T Luu ] [x,u] + [Lx,Lu].T [x,u].
        '''
        self.State = StateVector(nx)
        self.nx   = self.State.nx
        self.ndx  = self.State.ndx
        self.nu   = nu

        self.Lx  = None
        self.Lu  = None
        self.Lxx = None
        self.Lxu = None
        self.Luu = None
        self.Fx  = None
        self.Fu  = None

        self.unone = np.zeros(self.nu)
        
    def setUpRandom(self):
        self.Lx  = np.random.rand( self.ndx )
        self.Lu  = np.random.rand( self.nu  )
        self.Ls = L = np.random.rand( self.ndx+self.nu,self.ndx+self.nu )*2-1
        self.L = L = .5*np.dot(L.T,L)
        self.Lxx = L[:self.ndx,:self.ndx]
        self.Lxu = L[:self.ndx,self.ndx:]
        self.Luu = L[self.ndx:,self.ndx:]
        self.Fx  = np.random.rand( self.ndx,self.ndx )*2-1
        self.Fu  = np.random.rand( self.ndx,self.nu  )*2-1
        self.F   = np.random.rand( self.nx ) # Affine (nonautom) part of the dynamics
        
    def createData(self):
        return ActionDataLQR(self)

    def calc(model,data,x,u=None):
        '''Return xnext,cost for current state,control pair data.x,data.u. '''
        if u is None: u=model.unone
        quad = lambda a,Q,b: .5*np.dot(np.dot(Q,b).T,a)
        data.xnext = np.dot(model.Fx,x) + np.dot(model.Fu,u) + model.F
        data.cost  = quad(x,model.Lxx,x) + 2*quad(x,model.Lxu,u) + quad(u,model.Luu,u) \
                     + np.dot(model.Lx,x) + np.dot(model.Lu,u)
        return data.xnext,data.cost
        
    def calcDiff(model,data,x,u=None):
        if u is None: u=model.unone
        xnext,cost = model.calc(data,x,u)
        data.Lx  = model.Lx + np.dot(model.Lxx  ,x) + np.dot(model.Lxu,u)
        data.Lu  = model.Lu + np.dot(model.Lxu.T,x) + np.dot(model.Luu,u)
        data.Lxx = model.Lxx
        data.Lxu = model.Lxu
        data.Luu = model.Luu
        data.Fx  = model.Fx
        data.Fu  = model.Fu
        return xnext,cost



class ActionDataNumDiff:
    def __init__(self,model):
        nx,ndx,nu,ncost = model.nx,model.ndx,model.nu,model.ncost
        self.data0 = model.model0.createData()
        self.datax = [ model.model0.createData() for i in range(model.ndx) ]
        self.datau = [ model.model0.createData() for i in range(model.nu ) ]
        self.Lx = np.zeros([ model.ndx ])
        self.Lu = np.zeros([ model.nu ])
        self.Fx = np.zeros([ model.ndx,model.ndx ])
        self.Fu = np.zeros([ model.ndx,model.nu  ])
        if model.ncost >1 :
            self.Rx = np.zeros([model.ncost,model.ndx])
            self.Ru = np.zeros([model.ncost,model.nu ])
        if model.withGaussApprox:
            self. L = np.zeros([ ndx+nu, ndx+nu ])
            self.Lxx = self.L[:ndx,:ndx]
            self.Lxu = self.L[:ndx,ndx:]
            self.Lux = self.L[ndx:,:ndx]
            self.Luu = self.L[ndx:,ndx:]

class ActionModelNumDiff:
    def __init__(self,model,withGaussApprox=False):
        self.model0 = model
        self.nx = model.nx
        self.ndx = model.ndx
        self.nu = model.nu
        self.State = model.State
        self.disturbance = 1e-5
        try:            self.ncost = model.ncost
        except:         self.ncost = 1
        self.withGaussApprox = withGaussApprox
        assert( not self.withGaussApprox or self.ncost>1 )
        
    def createData(self):
        return ActionDataNumDiff(self)
    def calc(model,data,x,u): return model.model0.calc(data.data0,x,u)
    def calcDiff(model,data,x,u):
        xn0,c0 = model.calc(data,x,u)
        h = model.disturbance
        dist = lambda i,n,h: np.array([ h if ii==i else 0 for ii in range(n) ])
        Xint  = lambda x,dx: model.State.integrate(x,dx)
        Xdiff = lambda x1,x2: model.State.diff(x1,x2)
        for ix in range(model.ndx):
            xn,c = model.model0.calc(data.datax[ix],Xint(x,dist(ix,model.ndx,h)),u)
            data.Fx[:,ix] = Xdiff(xn0,xn)/h
            data.Lx[  ix] = (c-c0)/h
            if model.ncost>1: data.Rx[:,ix] = (data.datax[ix].costResiduals-data.data0.costResiduals)/h
        for iu in range(model.nu):
            xn,c = model.model0.calc(data.datau[iu],x,u+dist(iu,model.nu,h))
            data.Fu[:,iu] = Xdiff(xn0,xn)/h
            data.Lu[  iu] = (c-c0)/h
            if model.ncost>1: data.Ru[:,iu] = (data.datau[iu].costResiduals-data.data0.costResiduals)/h
        if model.withGaussApprox:
            data.Lxx[:,:] = np.dot(data.Rx.T,data.Rx)
            data.Lxu[:,:] = np.dot(data.Rx.T,data.Ru)
            data.Lux[:,:] = data.Lxu.T
            data.Luu[:,:] = np.dot(data.Ru.T,data.Ru)