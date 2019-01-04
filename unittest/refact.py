import numpy as np
from numpy import matrix
from numpy.linalg import norm, inv, pinv, eig
from pinocchio.utils import *
import scipy.linalg as scl
xnew=[0,'debug only']
np .random.seed     (220)

'''
Numpy convention.
Let's store vector as 1-d array and matrices as 2-d arrays. Multiplication is done by np.dot.
'''
def raiseIfNan(A,error=None):
    if error is None: error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A))>1e30):
        raise error
    
# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

class StateVector:
    '''
    Basic state class in cartesian space, represented by a vector.
    Tangent, integration and difference are straightforward.
    '''
    def __init__(self,nx):
        self.nx = nx
        self.ndx = nx
        
    def zero(self):
        '''Return a zero reference configuration. '''
        return np.zeros([self.nx])
    def rand(self):
        '''Return a random configuration. '''
        return np.random.rand(self.nx)
    def diff(self,x1,x2):
        '''
        Return the tangent vector representing the difference between x1 and x2,
        i.e. dx such that x1 [+] dx = x2.
        '''
        return x2-x1
    def integrate(self,x1,dx):
        '''
        Return x2 = x1 [+] dx.
        Warning: no timestep here, if integrating a velocity v during an interval dt, 
        set dx = v dt .
        '''
        return x1 + dx
    
    
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


# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

class ActionModelUnicycle:
    def __init__(self):
        '''
        Transition model is xnext(x,u) = Fx*x + Fu*x.
        Cost model is cost(x,u) = 1/2 [x,u].T [Lxx Lxu ; Lxu.T Luu ] [x,u] + [Lx,Lu].T [x,u].
        '''
        self.State = StateVector(3)
        self.nx   = self.State.nx
        self.ndx  = self.State.ndx
        self.nu   = 2
        self.ncost = 5
        
        self.dt   = .1
        self.costWeights = [ 1,.03 ]
        self.unone = np.zeros(self.nu)
        
    def createData(self):
        return ActionDataUnicycle(self)
    def calc(model,data,x,u=None):
        if u is None: u=model.unone
        assert(x.shape == (model.nx,) and u.shape == (model.nu,) )
        assert(data.xnext.shape == (model.nx,))
        assert(data.costResiduals.shape == (model.ncost,))
        v,w = u
        c,s = np.cos(x[2]),np.sin(x[2])
        dx = np.array([ v*c, v*s, w ])
        data.xnext[:] = [ x[0]+c*v*model.dt, x[1]+s*v*model.dt, x[2]+w*model.dt ]
        data.costResiduals[:3]  = model.costWeights[0]*x
        data.costResiduals[3:5] = model.costWeights[1]*u
        data.cost = .5* sum(data.costResiduals**2)
        return data.xnext,data.cost
    
    def calcDiff(model,data,x,u=None):
        if u is None: u=model.unone
        xnext,cost = model.calc(data,x,u)

        ### Cost derivatives
        data.L[:] = np.diag( [model.costWeights[0]]*model.nx + [model.costWeights[1]]*model.nu )
        data.Lx[:] = x * ([model.costWeights[0]**2]*model.nx )
        data.Lu[:] = u * ([model.costWeights[1]**2]*model.nu )
        np.fill_diagonal(data.Lxx,model.costWeights[0]**2)
        np.fill_diagonal(data.Luu,model.costWeights[1]**2)

        ### Dynamic derivatives
        c,s,dt = np.cos(x[2]),np.sin(x[2]),model.dt
        v,w = u
        data.Fx[:] = [ [ 1, 0, -s*v*dt ],
                       [ 0, 1,  c*v*dt ],
                       [ 0, 0,  1      ] ]
        data.Fu[:] = [ [ c*model.dt,       0  ],
                       [ s*model.dt,       0  ],
                       [          0, model.dt ] ]
        return xnext,cost
        
class ActionDataUnicycle:
    def __init__(self,model):
        nx,nu,ncost = model.nx,model.nu,model.ncost
        self. L = np.zeros([ nx+nu, nx+nu ])
        self. g = np.zeros([ nx+nu ])
        self. F = np.zeros([ nx,nx+nu ])

        self. cost  = np.nan
        self. xnext = np.zeros([ nx ])
        self. costResiduals = np.zeros([ ncost ])  # Might be use for numdiff (Gauss-Newton appox)
        
        self.Lxx = self.L[:nx,:nx]
        self.Lxu = self.L[:nx,nx:]
        self.Lux = self.L[nx:,:nx]
        self.Luu = self.L[nx:,nx:]
        self.Lx  = self.g[:nx]
        self.Lu  = self.g[nx:]
        self.Fx = self.F[:,:nx]
        self.Fu = self.F[:,nx:]

# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
from numpy import cos,sin,arctan2

class StateUnicycle:
    nx = 4
    ndx = 3
    def __init__(self):
        pass

    def zero(self):   return np.array([0.,0.,1.,0.]) # a,b,c,s
    def rand(self):
        a,b,th = np.random.rand(3)
        return np.array([a,b,cos(a),sin(a)])
    def diff(self,x1,x2):
        '''
        Return log(x1^-1 x2) = log(1M2).
        We might consider to return rather log(2M1), however log(1M2) seems
        more consistant with the Euclidean diff, i.e. log(1M2) = ^1 v_12
        while diff(p1,p2) = p2-p1 = O2-O1 = 10+02 = 12.
        '''
        # x1 = oM1 , x2 = oM2
        # dx = 1M2 = oM1^-1 oM2 = (R(th2-th1)  R(-th1) (p2-p1))
        #    = (c1 da + s1 db, -s1 da + c1 db, th2-th1)
        c1,s1 = x1[2:]
        c2,s2 = x2[2:]
        da,db = x2[:2] - x1[:2] 
        return np.array([ da*c1+db*s1, -da*s1+db*c1, arctan2(-s1*c2+c1*s2 ,c1*c2+s1*s2 ) ])
    def integrate(self,x,dx):
        '''
        x2 = oM1(x) 1M2(dx) =  oM1 exp(^1 v_dx) = oM_{x+dx}.
        dx as to be expressed in the tangent space at x (and not in the tangent space
        at 0).
        '''
        c1,s1 = x[2:]
        c2,s2 = cos(dx[2]),sin(dx[2])
        a,b = x[:2]
        da,db  = dx[:2]
        return np.array([ a+c1*da-s1*db, b+s1*da+c1*db, c1*c2-s1*s2, c1*s2+s1*c2 ])
    def Jdiff(self,x1,x2,firstsecond='both'):
        '''
        Return the "jacobian" i.e. the tangent map of diff with respect to the 
        first or the second variable. <firstsecond> should be 'first', 'second', or 'both'.
        If both, returns a tuple with the two maps.
        Jdiff1 = (diff(x1+dx,x2) - diff(x1,x2)) / dx.
        
        diff2(x) = diff(y,x) = log(y^-1x) = log(yx) with yx = y^-1 x = x in y frame.
        diff2(x+dx) = log(yx dx) = ( a + c da - s db, b + s da + c db, th+dth )
        with yx = a,b,th and dx = da,db,dth
        Jdiff2 = d(diff2)/d(dx) = [ c -s 0 ; s c 0 ; 0 0 1 ].
        '''
        assert(firstsecond in ['first', 'second', 'both' ])
        if firstsecond == 'both': return [ self.Jdiff(x1,x2,'first'),
                                           self.Jdiff(x1,x2,'second') ]
        a,b,th = self.diff(x1,x2); c,s = cos(th),sin(th)
        if firstsecond == 'second':
            return np.array([ [ c,-s,0 ],[ +s,c,0 ],[ 0,0,1 ] ])
        elif firstsecond == 'first': 
            return np.array([ [ -1,0,b ],[ 0,-1,-a ],[ 0,0,-1 ] ])
    def Jintegrate(self,x,v,firstsecond='both'):
        '''
        Return the "jacobian" i.e. the tangent map of diff with respect to the 
        first or the second variable. <firstsecond> should be 'first', 'second', or 'both'.
        If both, returns a tuple with the two maps.
        Ji1 = diff(int(x+dx,vx) - int(x,vx)) / dx.
        '''
        assert(firstsecond in ['first', 'second', 'both' ])
        if firstsecond == 'both': return [ self.Jintegrate(x,v,'first'),
                                           self.Jintegrate(x,v,'second') ]
        a,b,th = v; c,s = cos(th),sin(th)
        if firstsecond == 'second':
            return np.array([ [ c,s,0 ],[ -s,c,0 ],[ 0,0,1 ] ])
        elif firstsecond == 'first': 
            return np.array([ [ c,s,-c*b+s*a ],[ -s,c,s*b+c*a ],[ 0,0,1 ] ])
    
    #for debug
    @staticmethod
    def x2m(x):
        a,b,c,s = x
        return np.array([
            [ c,-s,a ],
            [ s, c,b ],
            [ 0, 0,1 ] ])
    @staticmethod
    def dx2m(x):
        a,b,th = x
        c,s = cos(th),sin(th)
        return np.array([
            [ c,-s,a ],
            [ s, c,b ],
            [ 0, 0,1 ] ])
    @staticmethod
    def m2x(m):
        return np.array([ m[0,2],m[1,2],m[0,0],m[1,0] ])
    @staticmethod
    def m2dx(m):
        return np.array([ m[0,2],m[1,2],arctan2(m[1,0],m[0,0]) ])

X = StateUnicycle()
x1 = X.rand()
x2 = X.rand()
dx = X.diff(x1,x2)
x  = X.integrate(x1,dx)
assert(norm(x-x2)<1e-9)

class StateNumDiff:
    '''
    From a norm state class, returns a class able to num diff. 
    '''
    def __init__(self,State):
        self.State = State
        self.nx  = State.nx
        self.ndx = State.ndx
        self.disturbance = 1e-6
    def zero(self): return self.State.zero()
    def rand(self): return self.State.rand()
    def diff(self,x1,x2): return self.State.diff(x1,x2)
    def integrate(self,x,dx): return self.State.integrate(x,dx)
    def Jdiff(self,x1,x2,firstsecond='both'):
        assert(firstsecond in ['first', 'second', 'both' ])
        if firstsecond == 'both': return [ self.Jdiff(x1,x2,'first'),
                                           self.Jdiff(x1,x2,'second') ]
        dx = np.zeros(self.ndx)
        h  = self.disturbance
        J  = np.zeros([self.ndx,self.ndx])
        d0 = self.diff(x1,x2)
        if firstsecond=='first':
            for k in range(self.ndx):
                dx[k]  = h
                J[:,k] = self.diff(self.integrate(x1,dx),x2)-d0
                dx[k]  = 0
        elif firstsecond=='second':
            for k in range(self.ndx):
                dx[k]  = h
                J[:,k] = self.diff(x1,self.integrate(x2,dx))-d0
                dx[k]  = 0
        J /= h
        return J
    def Jintegrate(self,x,vx,firstsecond='both'):
        assert(firstsecond in ['first', 'second', 'both' ])
        if firstsecond == 'both': return [ self.Jintegrate(x,vx,'first'),
                                           self.Jintegrate(x,vx,'second') ]
        dx = np.zeros(self.ndx)
        h  = self.disturbance
        J  = np.zeros([self.ndx,self.ndx])
        d0 = self.integrate(x,vx)
        if firstsecond=='first':
            for k in range(self.ndx):
                dx[k]  = h
                J[:,k] = self.diff(d0,self.integrate(self.integrate(x,dx),vx))
                dx[k]  = 0
        elif firstsecond=='second':
            for k in range(self.ndx):
                dx[k]  = h
                J[:,k] = self.diff(d0,self.integrate(x,vx+dx))
                dx[k]  = 0
        J /= h
        return J

x1 = X.zero()
x1 = X.rand()
x2 = X.zero()
h = 1e-6
dx = np.zeros(3); dx[1] = h
J = X.Jdiff(x1,x2,'second')
dd = (X.diff(x1,X.integrate(x2,dx)) - X.diff(x1,x2))/h

Xnum = StateNumDiff(StateUnicycle())
x1 = X.rand()
x2 = X.rand()
J1   ,J2    = X   .Jdiff(x1,x2)
Jnum1,Jnum2 = Xnum.Jdiff(x1,x2)
assert(norm(J1-Jnum1)<10*Xnum.disturbance)
assert(norm(J2-Jnum2)<10*Xnum.disturbance)

x  = X.rand()
vx = np.random.rand(X.ndx)
J1   ,J2    = X   .Jintegrate(x,vx)
Jnum1,Jnum2 = Xnum.Jintegrate(x,vx)
assert(norm(J1-Jnum1)<10*Xnum.disturbance)
assert(norm(J2-Jnum2)<10*Xnum.disturbance)

class ActionModelUnicycleVar:
    def __init__(self):
        '''
        Transition model is xnext(x,u) = Fx*x + Fu*x.
        Cost model is cost(x,u) = 1/2 [x,u].T [Lxx Lxu ; Lxu.T Luu ] [x,u] + [Lx,Lu].T [x,u].
        '''
        self.State = StateUnicycle()
        self.nx   = self.State.nx
        self.ndx  = self.State.ndx
        self.nu   = 2
        self.ncost = 5
        
        self.dt   = .1
        self.costWeights = [ 1,.03 ]
        self.unone = np.zeros(self.nu)
        self.xref = self.State.zero()
        
    def createData(self):
        return ActionDataUnicycleVar(self)
    def calc(model,data,x,u=None):
        if u is None: u=model.unone
        assert(x.shape == (model.nx,) and u.shape == (model.nu,) )
        assert(data.xnext.shape == (model.nx,))
        assert(data.costResiduals.shape == (model.ncost,))
        #v,w = u
        #c,s = x[2:]
        #dx = np.array([ v*c, v*s, w ])
        #c2,s2 = cos(w*model.dt),sin(w*model.dt)
        #data.xnext[:] = [ x[0]+c*v*model.dt, x[1]-s*v*model.dt, c*c2-s*s2, c*s2+s*c2 ]
        dx = np.array([u[0],0,u[1]])*model.dt
        data.xnext[:] = model.State.integrate(x,dx)
        data.costResiduals[:3]  = model.costWeights[0]*model.State.diff(model.xref,x)
        data.costResiduals[3:5] = model.costWeights[1]*u
        data.cost = .5* sum(data.costResiduals**2)
        return data.xnext,data.cost
    
    def calcDiff(model,data,x,u=None):
        if u is None: u=model.unone
        xnext,cost = model.calc(data,x,u)
        nx,ndx,nu = model.nx,model.ndx,model.nu
        
        ### Cost derivatives
        data.R[:ndx,:ndx] = model.costWeights[0] * model.State.Jdiff(model.xref,x,'second')
        data.R[ndx:,ndx:] = np.diag([ model.costWeights[1] ]*nu ) 
        data.g[:] = np.dot(data.R.T,data.costResiduals)
        data.L[:,:] = np.dot(data.R.T,data.R)

        ### Dynamic derivatives
        dx = np.array([u[0],0,u[1]])*model.dt
        Jx,Ju = model.State.Jintegrate(x,dx)
        data.Fx[:] = Jx
        data.Fu[:,0] = Ju[:,0]*model.dt
        data.Fu[:,1] = Ju[:,2]*model.dt

        return xnext,cost
        
class ActionDataUnicycleVar:
    def __init__(self,model):
        nx,ndx,nu,ncost = model.nx,model.ndx,model.nu,model.ncost
        self. L = np.zeros([ ndx+nu, ndx+nu ])
        self. g = np.zeros([ ndx+nu ])
        self. F = np.zeros([ ndx,ndx+nu ])
        self. R = np.zeros([ ncost,ndx+nu ]) # Residual jacobian
        
        self. cost  = np.nan
        self. xnext = np.zeros([ nx ])
        self. costResiduals = np.zeros([ ncost ])  # Might be use for numdiff (Gauss-Newton appox)

        self.Rx = self.R[:,:ndx]
        self.Ru = self.R[:,ndx:]
        self.Lxx = self.L[:ndx,:ndx]
        self.Lxu = self.L[:ndx,ndx:]
        self.Lux = self.L[ndx:,:ndx]
        self.Luu = self.L[ndx:,ndx:]
        self.Lx  = self.g[:ndx]
        self.Lu  = self.g[ndx:]
        self.Fx = self.F[:,:ndx]
        self.Fu = self.F[:,ndx:]


# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

class ActionModelNumDiff:
    def __init__(self,model,withGaussApprox=False):
        self.model0 = model
        self.nx = model.nx
        self.ndx = model.ndx
        self.nu = model.nu
        self.State = model.State
        self.disturbance = 1e-5
        self.ncost = model.ncost if 'ncost' in model.__dict__ else 1
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
            
if __name__ == '__main__':
    X=StateVector(3)
    for i in range(100):
        x1=X.rand(); x2=X.rand()
        dx= X.diff(x1,x2)
        assert( norm(X.integrate(x1,X.diff(x1,x2)) - x2 )<1e-6 )
    
    model = ActionModelUnicycle()
    data = model.createData()
    x = model.State.rand()
    u = np.random.rand(model.nu)

    #x[:] = [0,0,0]
    #u[:] = [1,2]
    model.calcDiff(data,x,u)
    
    mnum = ActionModelNumDiff(model,withGaussApprox=True)
    dnum = mnum.createData()
    mnum.calc(dnum,x,u)
    mnum.calcDiff(dnum,x,u)
    assert( norm(data.Fx-dnum.Fx) < 10*mnum.disturbance )
    assert( norm(data.Fu-dnum.Fu) < 10*mnum.disturbance )
    assert( norm(data.Lx-dnum.Lx) < 10*mnum.disturbance )
    assert( norm(data.Lu-dnum.Lu) < 10*mnum.disturbance )
    assert( norm(dnum.Lxx-data.Lxx) < 10*mnum.disturbance )
    assert( norm(dnum.Lxu-data.Lxu) < 10*mnum.disturbance )
    assert( norm(dnum.Luu-data.Luu) < 10*mnum.disturbance )


if __name__ == '__main__':
    # Numdiff unittest of unicycle manifold
    model = ActionModelUnicycleVar()
    data = model.createData()
    X = model.State

    x0 = model.State.rand()
    u0 = np.random.rand(model.nu)
    x1 = model.calc(data,x0,u0)[0]

    x1bis = X.m2x(np.dot(X.x2m(x0),X.dx2m([u0[0]*model.dt,0,u0[1]*model.dt])))
    assert(norm(x1bis-x1)<1e-9)

    model3 = ActionModelUnicycle()
    data3 = model3.createData()

    x = X.rand()
    u = np.random.rand(model.nu)
    x3 = X.diff(X.zero(),x)

    xnext ,cost  = model .calc(data ,x ,u)
    xnext3,cost3 = model3.calc(data3,x3,u)
    assert( norm( X.integrate(X.zero(),xnext3) - xnext) <1e-9)
    assert( abs(cost-cost3) <1e-9)

    model.calcDiff(data,x0,u0)

    mnum = ActionModelNumDiff(model,withGaussApprox=True)
    dnum = mnum.createData()

    model.calcDiff(data,x,u)
    mnum .calcDiff(dnum,x,u)

    dx = np.array([u[0],0,u[1]])

    def diff(m,d,x,u): m.calcDiff(d,x,u); return d.Fu
    da = lambda x,v,w: diff(model,data,x,np.array([v,w]))
    dn = lambda x,v,w: diff(mnum, dnum,x,np.array([v,w]))
    
    for k in ['Ru', 'Lxu', 'Fu', 'Fx', 'Rx', 'L', 'Lxx', 'Lux', 'Lu', 'Lx', 'Luu']:
        assert( norm( data.__dict__[k] - dnum.__dict__[k] ) < mnum.disturbance*10 )

# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

class ShootingProblem:
    def __init__(self,initialState,runningModels,terminalModel):
        self.T = len(runningModels)
        self.initialState = initialState
        self.runningModels = runningModels
        self.runningDatas  = [ m.createData() for m in runningModels ]
        self.terminalModel = terminalModel
        self.terminalData  = terminalModel.createData()

    def calc(self,xs,us):
        return sum([ m.calc(d,x,u)[1]
                     for m,d,x,u in zip(self.runningModels,self.runningDatas,xs[:-1],us)]) \
            + self.terminalModel.calc(self.terminalData,xs[-1])[1]
    def calcDiff(self,xs,us):
        '''
        Compute the cost-and-dynamics functions-and-derivatives
        along a given pair of trajectories xs (states) and us (controls).
        '''
        assert(len(xs)==self.T+1)
        assert(len(us)==self.T)
        for m,d,x,u in zip(self.runningModels,self.runningDatas,xs[:-1],us):
            m.calcDiff(d,x,u)
        self.terminalModel.calcDiff(self.terminalData,xs[-1])
        return sum([ d.cost for d in self.runningDatas+[self.terminalData] ])
        
    def rollout(self,us):
        '''
        For a given control trajectory, integrate the dynamics from self.initialState.
        '''
        xs = [ self.initialState ]
        for m,d,u in zip(self.runningModels,self.runningDatas,us):
            xs.append( m.calc(d,xs[-1],u)[0].copy() )
        return xs
        
class SolverKKT:
    def __init__(self,shootingProblem):
        self.problem = shootingProblem
    
        self.nx  = sum([ m.nx  for m in self.models() ])
        self.ndx = sum([ m.ndx for m in self.models() ])
        self.nu  = sum([ m.nu  for m in self.problem.runningModels ])
        
        self.kkt    = np.zeros([ 2*self.ndx+self.nu, 2*self.ndx+self.nu ])
        self.kktref = np.zeros(  2*self.ndx+self.nu )

        self.hess   = self.kkt[:self.ndx+self.nu,:self.ndx+self.nu]
        self.jac    = self.kkt[self.ndx+self.nu:,:self.ndx+self.nu]
        self.jacT   = self.kkt[:self.ndx+self.nu,self.ndx+self.nu:]
        self.grad   = self.kktref[:self.ndx+self.nu]
        self.cval   = self.kktref[self.ndx+self.nu:]

        self.Lxx    = self.hess[:self.ndx,:self.ndx]
        self.Lxu    = self.hess[:self.ndx,self.ndx:]
        self.Lux    = self.hess[self.ndx:,:self.ndx]
        self.Luu    = self.hess[self.ndx:,self.ndx:]
        self.Lx     = self.grad[:self.ndx]
        self.Lu     = self.grad[self.ndx:]
        self.Fx     = self.jac[:,:self.ndx]
        self.Fu     = self.jac[:,self.ndx:]

        self.alphas = [ 10**(-n) for n in range(7) ]
        self.th_acceptStep = .1
        self.th_stop = 1e-9
        self.th_grad = 1e-12

        self.x_reg = 0
        self.u_reg = 0
        
    def models(self) : return self.problem.runningModels + [self.problem.terminalModel]
    def datas(self) : return self.problem.runningDatas + [self.problem.terminalData]
    def setCandidate(self,xs=None,us=None,isFeasible=False,copy=True):
        '''
        Set the solver candidate value for the decision variables, as a trajectory xs,us
        of T+1 and T elements. isFeasible should be set to True if the xs are 
        obtained from integrating the us (roll-out).
        If copy is True, make a copy of the data.
        '''
        if xs is None: xs = [ m.State.zero() for m in self.models() ]
        elif copy:     xs = [ x.copy() for x in xs ]
        if us is None: us = [ np.zeros(m.nu) for m in self.problem.runningModels ]
        elif copy:     us = [ u.copy() for u in us ]

        assert( len(xs) == self.problem.T+1 )
        assert( len(us) == self.problem.T )
        self.xs = xs
        self.us = us
        self.isFeasible = isFeasible

    def calc(self):
        '''
        For a given pair of candidate state and control trajectories,
        compute the KKT system. 
        isFeasible should be set True if F(xs,us) = 0 (xs obtained from rollout).
        '''
        xs,us = self.xs,self.us
        problem = self.problem
        self.cost = problem.calcDiff(xs,us)
        ix = 0; iu = 0
        cx0 = problem.runningModels[0].ndx  # offset on constraint xnext = f(x,u) due to x0 = ref.

        np.fill_diagonal(self.Fx,1)
        for model,data,xguess in zip(problem.runningModels,problem.runningDatas,xs[1:]):
            ndx = model.ndx; nu = model.nu
            self.Lxx[ix:ix+ndx,ix:ix+ndx] = data.Lxx
            self.Lxu[ix:ix+ndx,iu:iu+nu ] = data.Lxu
            self.Lux[iu:iu+nu ,ix:ix+ndx] = data.Lxu.T
            self.Luu[iu:iu+nu ,iu:iu+nu ] = data.Luu

            self.Lx[ix:ix+ndx]           = data.Lx
            self.Lu[iu:iu+nu ]           = data.Lu
            
            self.Fx[cx0+ix:cx0+ix+ndx,ix:ix+ndx] = - data.Fx
            self.Fu[cx0+ix:cx0+ix+ndx,iu:iu+nu ] = - data.Fu

            # constraint value = xnext_guess - f(x_guess,u_guess) = diff(f,xnext_guesss)
            self.cval[cx0+ix:cx0+ix+ndx] = model.State.diff(data.xnext,xguess) #data.F
            ix += ndx; iu += nu
            
        model,data = problem.terminalModel,problem.terminalData
        ndx = model.ndx; nu = model.nu
        self.Lxx[ix:ix+ndx,ix:ix+ndx] = data.Lxx
        self.Lx [ix:ix+ndx]           = data.Lx

        # constraint value = x_guess - x_ref = diff(x_ref,x_guess)
        self.cval[:cx0] = problem.runningModels[0].State.diff(problem.initialState,xs[0])

        self.jacT[:] = self.jac.T

        # Regularization
        ndx = sum([ m.ndx for m in self.models() ])
        nu  = sum([ m.nu  for m in self.problem.runningModels ])
        if self.x_reg != 0: self.kkt[range(ndx),       range(ndx)       ] += self.x_reg
        if self.u_reg != 0: self.kkt[range(ndx,ndx+nu),range(ndx,ndx+nu)] += self.u_reg
                
        return self.cost
    
    def computeDirection(self,recalc=True):
        '''
        Compute the direction of descent for the current guess xs,us. 
        if recalc is True, run self.calc() before hand. self.setCandidate 
        must have been called before.
        '''
        if recalc: self.calc()
        self.primaldual = np.linalg.solve(self.kkt,-self.kktref)
        self.primal = self.primaldual[:self.ndx+self.nu]
        p_x  = self.primaldual[:self.ndx]
        p_u  = self.primaldual[self.ndx:self.ndx+self.nu]
        self.dual = self.primaldual[-self.ndx:]
        ix = 0; iu = 0
        dxs = []; dus = []; lambdas = []
        for model,data in zip(self.problem.runningModels,self.problem.runningDatas):
            ndx = model.ndx; nu = model.nu
            dxs.append( p_x[ix:ix+ndx] )
            dus.append( p_u[iu:iu+nu] )
            lambdas.append( self.dual[ix:ix+ndx] )
            ix+=ndx; iu+=nu
        dxs.append( p_x[-self.problem.terminalModel.ndx:] )
        lambdas.append( self.dual[-self.problem.terminalModel.ndx:] )
        self.dxs = dxs ; self.dus = dus ; self.lambdas = lambdas
        return dxs,dus,lambdas

    def expectedImprovement(self):
        '''
        Return the expected improvement that the current step should bring,
        i.e dV_exp = f(xk) - f_exp(xk + delta), where f_exp(xk+delta) = f_k + m_k(delta)
        and m_k is the quadratic model of f at x_k, and delta is the descent direction.
        Then: dv_exp = - grad*delta - .5*hess*delta**2
        '''
        return -np.dot(self.grad,self.primal), \
            -np.dot(np.dot(self.hess,self.primal),self.primal)

    def stoppingCriteria(self):
        '''
        Return two terms:
        * sumsq(dL+dF) = || d/dx Cost - d/dx (lambda^T constraint) ||^2
        * sumsq(cval)  = || d/dlambda L ||^2 = || constraint ||^2
        '''
        lambdas = self.lambdas
        dL = self.grad
        dF = np.concatenate([ lk - np.dot(lkp1,dk.Fx)
                              for lk,dk,lkp1 in zip( lambdas[:-1],
                                                     self.problem.runningDatas,
                                                     lambdas[1:]) ] \
                            + [ lambdas[-1] ] \
                            + [ -np.dot(lkp1,dk.Fu)
                                for lk,dk,lkp1 in zip( lambdas[:-1],
                                                       self.problem.runningDatas,
                                                       lambdas[1:]) ])
        return sum((dL+dF)**2), sum(self.cval**2)
        
    def tryStep(self,stepLength):
        '''
        Compute in x_try,u_try a step by adding stepLength*dx to x.
        Store the result in self.xs_try,self.us_try.
        Return the cost improvement ie cost(xs,us)-cost(xs_try,us_try).
        '''
        self.xs_try = [ m.State.integrate(x,stepLength*dx)
                        for m,x,dx in zip(self.models(),self.xs,self.dxs) ]
        self.us_try = [ u+stepLength*du for u,du in zip(self.us,self.dus) ]
        self.cost_try = self.problem.calc(self.xs_try,self.us_try)
        dV  = self.cost-self.cost_try
        return dV
    
    def solve(self,maxiter=100,init_xs = None,init_us = None,isFeasible=False,verbose=False):
        '''
        From an initial guess init_xs,init_us (feasible or not),
        iterate over computeDirection and tryStep until stoppingCriteria is below threshold.
        '''
        self.setCandidate(init_xs,init_us,isFeasible,copy=True)
        for i in range(maxiter):
            self.computeDirection()
            d1,d2 = self.expectedImprovement()

            for a in self.alphas:
                dV = self.tryStep(a)
                if verbose: print('\t\tAccept? %f %f' % (dV, d1*a+.5*d2*a**2) )
                if d1<1e-9 or not isFeasible or dV > self.th_acceptStep*(d1*a+.5*d2*a**2):
                    # Accept step
                    self.setCandidate(self.xs_try,self.us_try,isFeasible=True,copy=False)
                    break
            if verbose: print( 'Accept a=%f, cost=%.8f' % (a,self.problem.calc(self.xs,self.us)))
                
            self.stop = sum(self.stoppingCriteria())
            if self.stop<self.th_stop:
                return self.xs,self.us,True
            # if d1<self.th_grad:
            #     return self.xs,self.us,False

        # Warning: no convergence in max iterations
        return self.xs,self.us,False
                
# ---------------------------------------------------
# --- TEST ------------------------------------------
# ---------------------------------------------------

NX = 1
NU = 1
X = StateVector(NX)
assert(X.zero().shape == (NX,))
assert(X.rand().shape == (NX,))

x1 = X.rand()
x2 = X.rand()
assert( np.linalg.norm(x2-X.integrate( x1,X.diff(x1,x2) )) < 1e-9 )

# ---------------------------------------------------
model = ActionModelUnicycle()
#model = ActionModelLQR(NX,NU); model.setUpRandom()
data  = model.createData()
LQR = isinstance(model,ActionModelLQR)

x = model.State.rand()
u = np.zeros([model.nu])

# Check dimensions of calc and calcdiff
assert( model.calc(data,x,u)[0].shape == (model.nx,) )
assert( isinstance(model.calc(data,x,u)[1],float) )
assert( model.calcDiff(data,x,u)[0].shape == (model.nx,) )

# Check dimension of KKT.
problem = ShootingProblem(model.State.zero()+1, [ model, model ], model)
kkt = SolverKKT(problem)
xs = [ m.State.zero()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.zeros(m.nu)       for m in problem.runningModels ]
assert(len(np.concatenate(xs[:-1])) == sum([ m.nx for m in problem.runningModels]) )
assert(len(np.concatenate(us))      == sum([ m.nu for m in problem.runningModels]) )

# Test dimensions of KKT calc.
kkt.setCandidate(xs,us)
kkt.calc()
assert( np.linalg.norm(kkt.hess-kkt.hess.T)<1e-9)
assert(len(filter(lambda x:x>0,eig(kkt.kkt)[0]))==kkt.nx+kkt.nu)
assert(len(filter(lambda x:x<0,eig(kkt.kkt)[0]))==kkt.nx)

# Test that the solution respect the dynamics (or linear approx of it).
dxs,dus,ls = kkt.computeDirection()
x0,x1,x2 = dxs
u0,u1 = dus
l0,l1,l2 = ls
assert(np.linalg.norm(dxs[0]-problem.initialState)<1e-9 )
if LQR: assert(np.linalg.norm(model.calc(data,x0,u0)[0]-x1)<1e-9)
for i,_ in enumerate(dus):
    h = 1 if LQR else 1e-6
    assert(np.linalg.norm(model.calc(data,dxs[i]*h,dus[i]*h)[0]/h-dxs[i+1])<10*h)

# If LQR. test that a random solution respect the dynamics
xs = [ m.State.rand()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.random.rand(m.nu) for m in problem.runningModels ]
kkt.setCandidate(xs,us)
dxs,dus,ls = kkt.computeDirection()
assert(np.linalg.norm(xs[0]+dxs[0]-problem.initialState)<1e-9 )
for i,_ in enumerate(dus):
    if LQR:
        assert(np.linalg.norm(model.calc(data,xs[i]+dxs[i],us[i]+dus[i])[0]-(xs[i+1]+dxs[i+1]))<1e-9)

# Test the optimality of the QP solution (underlying KKT).
quad = lambda a,Q,b: .5*np.dot(np.dot(Q,b).T,a)
cost = np.dot(.5*np.dot(kkt.hess,kkt.primal)+kkt.grad,kkt.primal)
for i in range(1000):
    eps=1e-1*np.random.rand(*kkt.primal.shape)
    eps -= np.dot(np.dot(np.linalg.pinv(kkt.jac),kkt.jac),eps) # project eps in null(jac)
    assert( cost<=np.dot(.5*np.dot(kkt.hess,kkt.primal+eps)+kkt.grad,kkt.primal+eps) )

# Check improvement model
_,_,done = kkt.solve(maxiter=0)
assert(not done)
x0s,u0s = kkt.xs,kkt.us
kkt.setCandidate(x0s,u0s)
dxs,dus,ls = kkt.computeDirection()
dv = kkt.tryStep(1)
x1s = [ x+dx for x,dx in zip(x0s,dxs) ]
u1s = [ u+du for u,du in zip(u0s,dus) ]
for xt,x1 in zip(kkt.xs_try,x1s): assert( norm(xt-x1) < 1e-9 )
for ut,u1 in zip(kkt.us_try,u1s): assert( norm(ut-u1) < 1e-9 )
d1,d2 = kkt.expectedImprovement()
if LQR: assert( d1+d2/2 + problem.calc(x1s,u1s) < 1e-9 )
assert( d1+d2/2 + np.dot(.5*np.dot(kkt.hess,np.concatenate(dxs+dus))+kkt.grad,
                         np.concatenate(dxs+dus)) < 1e-9 )

# Check stoping criteria
kkt.setCandidate(x1s,u1s)
kkt.calc()
dL,dF=kkt.stoppingCriteria()
if LQR: assert( dL+dF < 1e-9 )
assert( abs(kkt.stoppingCriteria()[0] -sum((kkt.grad+np.dot(kkt.jacT,kkt.dual))**2))<1e-9)

xopt,uopt,done = kkt.solve(maxiter=200)
assert(done)
for i,_ in enumerate(uopt):
    assert(np.linalg.norm(model.calc(data,xopt[i],uopt[i])[0]-xopt[i+1])<1e-9)

###  INTEGRATIVE TEST ###
###  INTEGRATIVE TEST ###
###  INTEGRATIVE TEST ###
T = 10
WITH_PLOT = not True 

def disp(x):
    sc,delta = .1,.05
    a,b,th = x[:3]; c,s=np.cos(th),np.sin(th)
    refs = []
    refs.append(plt.arrow(a-sc/2*c-delta*s,b-sc/2*s+delta*c,c*sc,s*sc,head_width=.03))
    refs.append(plt.arrow(a-sc/2*c+delta*s,b-sc/2*s-delta*c,c*sc,s*sc,head_width=.03))
    return refs

runcost  = ActionModelUnicycle(); runcost .costWeights[1] = 10
termcost = ActionModelUnicycle(); termcost.costWeights[0] = 1000
problem = ShootingProblem(np.array([1,0,3]), [ runcost, ]*T, termcost)
kkt = SolverKKT(problem)
xs,us,done = kkt.solve()
assert(norm(xs[-1])<1e-2)

if WITH_PLOT:
    import matplotlib.pyplot as plt
    for x in xs: disp(x)
    ax = max(np.concatenate([ (abs(x[0]),abs(x[1])) for x in xs]))*1.2
    plt.axis([-ax,ax,-ax,ax])
    plt.show()


# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

class SolverAbstract:
    def __init__(self,shootingProblem):
        self.problem = shootingProblem
    
        self.alphas = [ 10**(-n) for n in range(7) ]
        self.th_acceptStep = .1
        self.th_stop = 1e-12
        
    def models(self) : return self.problem.runningModels + [self.problem.terminalModel]
    def datas(self) : return self.problem.runningDatas + [self.problem.terminalData]

    def setCandidate(self,xs=None,us=None,isFeasible=False,copy=True):
        '''
        Set a candidate pair xs,us, and define it as feasible (i.e. obtained from
        rollout) or not.
        '''
        pass
    def calc(self):
        '''
        Compute the tangent (LQR) model.
        Returns cost.
        '''
        self.cost = problem.calcDiff(xs,us)
        return self.cost
        
    def computeDirection(self,recalc=True):
        '''
        Compute the descent direction dx,dx.
        Returns the descent direction dx,du and the dual lambdas as lists of T+1, T and T+1 lengths. 
        '''
        if recalc: self.calc()
        return # xs,us,lambdas

    def tryStep(self,stepLength):
        '''
        Make a step of length stepLength in the direction self.dxs,self.dus computed 
        (possibly partially) by computeDirection. Store the result in self.xs_try,self.us_try.
        Return the cost improvement, i.e. self.cost-self.problem.calc(self.xs_try,self.us_try).
        '''
        # self.xs_try,self.us_try = ...
        # dV = self.cost - ...
        return # dV
            
    def stoppingCriteria(self):
        '''
        Return a list of positive parameters whose sum quantifies the algorithm termination.
        '''
        return    #[ criterion1, criterion2 ... ]
    
    def expectedImprovement(self):
        '''
        Return two scalars denoting the quadratic improvement model
        (i.e. dV = f_0 - f_+ = d1*a + d2*a**2/2)
        '''
        return # [ d1, d2 ]

    def solver(self,maxiter=100,init_xs=None,init_us=None):
        '''
        Nonlinear solver iterating over the solveQP.
        Return the optimum xopt,uopt as lists of T+1 and T terms, and a boolean
        describing the success.
        '''

from itertools import izip
rev_enumerate = lambda l: izip(xrange(len(l)-1, -1, -1), reversed(l))

class SolverDDP:
    def __init__(self,shootingProblem):
        self.problem = shootingProblem
        self.allocate()

        self.isFeasible = False  # Change it to true if you know that datas[t].xnext = xs[t+1]
        self.alphas = [ 4**(-n) for n in range(10) ]
        self.th_acceptStep = .1
        self.th_stop = 1e-9
        self.th_grad = 1e-12

        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5
        
    def models(self) : return self.problem.runningModels + [self.problem.terminalModel]
    def datas(self) : return self.problem.runningDatas + [self.problem.terminalData]

    def setCandidate(self,xs=None,us=None,isFeasible=False,copy=True):
        '''
        Set the solver candidate value for the decision variables, as a trajectory xs,us
        of T+1 and T elements. isFeasible should be set to True if the xs are 
        obtained from integrating the us (roll-out).
        If copy is True, make a copy of the data.
        '''
        if xs is None: xs = [ m.State.zero() for m in self.models() ]
        elif copy:     xs = [ x.copy() for x in xs ]
        if us is None: us = [ np.zeros(m.nu) for m in self.problem.runningModels ]
        elif copy:     us = [ u.copy() for u in us ]

        assert( len(xs) == self.problem.T+1 )
        assert( len(us) == self.problem.T )
        self.xs = xs
        self.us = us
        self.isFeasible = isFeasible

    def calc(self):
        '''
        Compute the tangent (LQR) model.
        Returns nothing.
        '''
        self.cost = self.problem.calcDiff(self.xs,self.us)
        return self.cost
    
    def computeDirection(self,recalc=True):
        '''
        Compute the descent direction dx,dx.
        Returns the descent direction dx,du and the dual lambdas as lists of T+1, T and T+1 lengths. 
        '''
        if recalc: self.calc()
        self.backwardPass()
        return [ np.nan ]*(self.problem.T+1), self.k, self.Vx
        
    def stoppingCriteria(self):
        '''
        Return a sum of positive parameters whose sum quantifies the algorithm termination.
        '''
        return  [ sum(q**2) for q in self.Qu ]
    
    def expectedImprovement(self):
        '''
        Return two scalars denoting the quadratic improvement model
        (i.e. dV = f_0 - f_+ = d1*a + d2*a**2/2)
        '''
        d1 = sum([  np.dot(q,k)           for q,k in zip(self.Qu,self.k) ])
        d2 = sum([ -np.dot(k,np.dot(q,k)) for q,k in zip(self.Quu,self.k) ])
        return [ d1, d2 ]

    def tryStep(self,stepLength):
        self.forwardPass(stepLength)
        return self.cost - self.cost_try
    
    def solve(self,maxiter=100,init_xs=None,init_us=None,isFeasible=False,verbose=False,regInit=None):
        '''
        Nonlinear solver iterating over the solveQP.
        Return the optimum xopt,uopt as lists of T+1 and T terms, and a boolean
        describing the success.
        '''
        self.setCandidate(init_xs,init_us,isFeasible=isFeasible,copy=True)
        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin
        
        for i in range(maxiter):
            try:
                self.computeDirection()
            except ArithmeticError:
                if verbose: print ('Backward pass failed')
                self.increaseRegularization()
            d1,d2 = self.expectedImprovement()

            for a in self.alphas:
                try:
                    if verbose: print ('Forward pass failed')
                    dV = self.tryStep(a)
                except ArithmeticError:
                    continue
                dV_exp = a*(d1+.5*d2*a)
                if verbose: print('\t\tAccept? %f %f' % (dV, d1*a+.5*d2*a**2) )
                if d1<self.th_grad or not self.isFeasible or dV > self.th_acceptStep*dV_exp:
                    # Accept step
                    self.setCandidate(self.xs_try,self.us_try,isFeasible=True,copy=False)
                    self.cost = self.cost_try
                    break
            if verbose: print( 'Accept a=%f, cost=%.8f' % (a,self.problem.calc(self.xs,self.us)))
            if a>self.th_step: self.decreaseRegularization()
            
            self.stop = sum(self.stoppingCriteria())
            if self.stop<self.th_stop:
                return self.xs,self.us,True
            # if d1<self.th_grad:
            #     return self.xs,self.us,False
            
        # Warning: no convergence in max iterations
        return self.xs,self.us,False

    def increaseRegularization(self):
        self.x_reg *= self.regFactor
        if self.x_reg > self.regMax: self.x_reg = self.regMax
        self.u_reg = self.x_reg
    def decreaseRegularization(self):
        self.x_reg /= self.regFactor
        if self.x_reg < self.regMin: self.x_reg = self.regMin
        self.u_reg = self.x_reg

    
    #### DDP Specific
    def allocate(self):
        '''
        Allocate matrix space of Q,V and K. 
        Done at init time (redo if problem change).
        '''
        self.Vxx = [ np.zeros([m.nx     ,m.nx     ]) for m in self.models() ]
        self.Vx  = [ np.zeros([m.nx])                for m in self.models() ]

        self.Q   = [ np.zeros([m.nx+m.nu,m.nx+m.nu]) for m in self.problem.runningModels ]
        self.q   = [ np.zeros([m.nx+m.nu          ]) for m in self.problem.runningModels ]
        self.Qxx = [ Q[:m.nx,:m.nx] for m,Q in zip(self.problem.runningModels,self.Q) ]
        self.Qxu = [ Q[:m.nx,m.nx:] for m,Q in zip(self.problem.runningModels,self.Q) ]
        self.Qux = [ Qxu.T          for m,Qxu in zip(self.problem.runningModels,self.Qxu) ]
        self.Quu = [ Q[m.nx:,m.nx:] for m,Q in zip(self.problem.runningModels,self.Q) ]
        self.Qx  = [ q[:m.nx]       for m,q in zip(self.problem.runningModels,self.q) ]
        self.Qu  = [ q[m.nx:]       for m,q in zip(self.problem.runningModels,self.q) ]

        self.K   = [ np.zeros([ m.nu,m.nx ]) for m in self.problem.runningModels ]
        self.k   = [ np.zeros([ m.nu      ]) for m in self.problem.runningModels ]
        
    def backwardPass(self):
        xs,us = self.xs,self.us
        self.Vx [-1][:]   = self.problem.terminalData.Lx
        self.Vxx[-1][:,:] = self.problem.terminalData.Lxx
        if self.x_reg != 0:
            ndx = self.problem.terminalModel.ndx
            self.Vxx[-1][range(ndx),range(ndx)] += self.x_reg
        
        for t,(model,data) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            self.Qxx[t][:,:] = data.Lxx + np.dot(data.Fx.T,np.dot(self.Vxx[t+1],data.Fx))
            self.Qxu[t][:,:] = data.Lxu + np.dot(data.Fx.T,np.dot(self.Vxx[t+1],data.Fu))
            self.Quu[t][:,:] = data.Luu + np.dot(data.Fu.T,np.dot(self.Vxx[t+1],data.Fu))
            self.Qx [t][:]   = data.Lx  + np.dot(self.Vx[t+1],data.Fx)
            self.Qu [t][:]   = data.Lu  + np.dot(self.Vx[t+1],data.Fu)
            if not self.isFeasible:
                # In case the xt+1 are not f(xt,ut) i.e warm start not obtained from roll-out.
                relinearization = np.dot(self.Vxx[t+1],model.State.diff(xs[t+1],data.xnext))
                self.Qx [t][:] += np.dot(data.Fx.T,relinearization)
                self.Qu [t][:] += np.dot(data.Fu.T,relinearization)

            if self.u_reg != 0: self.Quu[t][range(model.nu),range(model.nu)] += self.u_reg

            try:
                Lb = scl.cho_factor(self.Quu[t])
                self.K[t][:,:]   = scl.cho_solve(Lb,self.Qux[t])
                self.k[t][:]     = scl.cho_solve(Lb,self.Qu [t])
            except scl.LinAlgError:
                raise ArithmeticError('backward error')
                
            # Vx = Qx - Qu K + .5(- Qxu k - k Qux + k Quu K + K Quu k)
            # Qxu k = Qxu Quu^+ Qu
            # Qu  K = Qu Quu^+ Qux = Qxu k
            # k Quu K = Qu Quu^+ Quu Quu^+ Qux = Qu Quu^+ Qux if Quu^+ = Quu^-1
            if self.u_reg == 0:
                self.Vx[t][:] = self.Qx [t] - np.dot(self.Qu [t],self.K[t])
            else:
                self.Vx[t][:] = self.Qx [t] - 2*np.dot(self.Qu [t],self.K[t]) \
                                + np.dot(np.dot(self.k[t],self.Quu[t]),self.K[t])
            self.Vxx[t][:,:] = self.Qxx[t] - np.dot(self.Qxu[t],self.K[t])

            if self.x_reg != 0: self.Vxx[t][range(model.ndx),range(model.ndx)] += self.x_reg
            raiseIfNan(self.Vxx[t],ArithmeticError('backward error'))
            raiseIfNan(self.Vx[t],ArithmeticError('backward error'))
            
    def forwardPass(self,stepLength,b=None,warning='ignore'):
        # Argument b is introduce for debug purpose.
        # Argument warning is also introduce for debug: by default, it masks the numpy warnings
        #    that can be reactivated during debug.
        if b is None: b=1
        xs,us = self.xs,self.us
        xtry = [ self.problem.initialState ] + [ np.nan ]*self.problem.T
        utry = [ np.nan ]*self.problem.T
        ctry = 0
        for t,(m,d) in enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            utry[t] = us[t] - self.k[t]*stepLength  \
                      - np.dot(self.K[t],m.State.diff(xs[t],xtry[t]))*b
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                xnext,cost = m.calc(d, xtry[t],utry[t] )
            xtry[t+1] = xnext.copy()  # not sure copy helpful here.
            ctry += cost
            raiseIfNan([ctry,cost],ArithmeticError('forward error'))
            raiseIfNan(xtry[t+1],ArithmeticError('forward error'))
        with np.warnings.catch_warnings() as npwarn:
            np.warnings.simplefilter(warning)
            ctry += self.problem.terminalModel.calc(self.problem.terminalData,xtry[-1])[1]
        raiseIfNan(ctry,ArithmeticError('forward error'))
        self.xs_try = xtry ; self.us_try = utry; self.cost_try = ctry
        return xtry,utry,ctry
            
# --- TEST DDP ---
model = ActionModelLQR(1,1); model.setUpRandom()
#model = ActionModelUnicycle()
nx,nu = model.nx,model.nu

problem = ShootingProblem(model.State.zero()+1, [ model ], model)
ddp = SolverDDP(problem)

xs = [ m.State.zero()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.zeros(m.nu)       for m in problem.runningModels ]
#xs[0][:] = problem.initialState
xs[0] = np.random.rand(nx)
us[0] = np.random.rand(nu)
xs[1] = np.random.rand(nx)


ddp.setCandidate(xs,us)
ddp.computeDirection()
xnew,unew,cnew = ddp.forwardPass(stepLength=1)

# Check versus simple (1-step) DDP
ddp.problem.calcDiff(xs,us)
l0x  = problem.runningDatas[0].Lx
l0u  = problem.runningDatas[0].Lu
l0xx = problem.runningDatas[0].Lxx
l0xu = problem.runningDatas[0].Lxu
l0uu = problem.runningDatas[0].Luu
f0x  = problem.runningDatas[0].Fx
f0u  = problem.runningDatas[0].Fu
x1pred = problem.runningDatas[0].xnext

v1x  = problem.terminalData.Lx
v1xx = problem.terminalData.Lxx

relin1 = np.dot(v1xx,x1pred-xs[1])
q0x  = l0x  + np.dot(f0x.T,v1x) + np.dot(f0x.T,relin1)
q0u  = l0u  + np.dot(f0u.T,v1x) + np.dot(f0u.T,relin1)
q0xx = l0xx + np.dot(f0x.T,np.dot(v1xx,f0x))
q0xu = l0xu + np.dot(f0x.T,np.dot(v1xx,f0u))
q0uu = l0uu + np.dot(f0u.T,np.dot(v1xx,f0u))

K0 = np.dot(inv(q0uu),q0xu.T)
k0 = np.dot(inv(q0uu),q0u)
assert(norm(K0-ddp.K[0])<1e-9)
assert(norm(k0-ddp.k[0])<1e-9)

v0x  = q0x  - np.dot(q0xu,k0)
v0xx = q0xx - np.dot(q0xu,K0)
assert(norm(v0xx-ddp.Vxx[0])<1e-9)

x0 = problem.initialState
u0 = us[0] -k0 - np.dot(K0,x0-xs[0])
x1 = model.calc(data,x0,u0)[0]

assert(norm(unew[0]-u0)<1e-9)
assert(norm(xnew[1]-x1)<1e-9)


x0ref = problem.initialState
f0 = x1pred
l1x = v1x
l1xx = v1xx

#for n in [ 'x0ref','l0x','l0u','l0xx','l0xu','l0uu','f0x','f0u','f0','l1x','l1xx' ]:
#    print(n[0]+n[2:]+'['+n[1]+'] : '+str(float(locals()[n])))



# --- TEST DDP vs KKT LQR ---
np .random.seed     (220)
model = ActionModelLQR(3,3); model.setUpRandom();
#model = ActionModelUnicycle()
nx = model.nx
nu = model.nu
T = 1

problem = ShootingProblem(model.State.zero()+2, [ model ]*T, model)
xs = [ m.State.zero()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.zeros(m.nu)       for m in problem.runningModels ]
x0ref = problem.initialState
xs[0] = np.random.rand(nx)
xs[1] = np.random.rand(nx)
us[0] = np.random.rand(nu)
#xs[1] = model.calc(data,xs[0],us[0])[0].copy()

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

kkt.setCandidate(xs,us)
dxkkt,dukkt,lkkt = kkt.computeDirection()
xkkt,ukkt,donekkt = kkt.solve(maxiter=2)
assert(donekkt)

ddp.setCandidate(xs,us)
ddp.computeDirection()
xddp,uddp,costddp = ddp.forwardPass(stepLength=1)
assert( norm(xddp[0]-xkkt[0])<1e-9 )
assert( norm(xddp[1]-xkkt[1])<1e-9 )
assert( norm(uddp[0]-ukkt[0])<1e-9 )

# Test step length
us = [ np.random.rand(m.nu) for m in problem.runningModels ]
xs = problem.rollout(us)
kkt.setCandidate(xs,us,isFeasible=True)
ddp.setCandidate(xs,us,isFeasible=False)
dxkkt,dukkt,lkkt = kkt.computeDirection()
step = .1
dvkkt = kkt.tryStep(step); xkkt,ukkt = kkt.xs_try,kkt.us_try
dxddp,duddp,lddp = ddp.computeDirection()
dvddp = ddp.tryStep(step); xddp,uddp = ddp.xs_try,ddp.us_try

assert( norm(xddp[0]-xkkt[0])<1e-9 )
assert( norm(xddp[1]-xkkt[1])<1e-9 )
assert( norm(uddp[0]-ukkt[0])<1e-9 )

d1,d2 = kkt.expectedImprovement()
assert( abs(dvkkt-d1*step-.5*d2*step**2) < 1e-9 )

dd1,dd2 = ddp.expectedImprovement()
assert( abs(d1-dd1)<1e-9 and abs(d2-dd2)<1e-9 )

# Test stopping criteria at optimum
ddp.tryStep(1)
ddp.setCandidate(ddp.xs_try,ddp.us_try)
kkt.setCandidate(ddp.xs_try,ddp.us_try)
ddp.computeDirection()
kkt.computeDirection()
assert(sum(ddp.stoppingCriteria())<1e-9)
assert(sum(kkt.stoppingCriteria())<1e-9)

# --- TEST DDP VS KKT NLP ---
model = ActionModelUnicycle()
nx = model.nx
nu = model.nu
T = 1

problem = ShootingProblem(model.State.zero()+2, [ model ]*T, model)
xs = [ m.State.zero()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.zeros(m.nu)       for m in problem.runningModels ]
x0ref = problem.initialState
xs[0] = np.random.rand(nx)
us[0] = np.random.rand(nu)
xs[1] = np.random.rand(nx)
#xs[1] = model.calc(data,xs[0],us[0])[0].copy()

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

kkt.setCandidate(xs,us)
dxkkt,dukkt,lkkt = kkt.computeDirection()
xkkt = [ x+dx for x,dx in zip(xs,dxkkt) ]
ukkt = [ u+du for u,du in zip(us,dukkt) ]

ddp.setCandidate(xs,us)
ddp.computeDirection()
xddp,uddp,costddp = ddp.forwardPass(stepLength=1)
assert( norm(xddp[0]-xkkt[0])<1e-9 )
assert( norm(uddp[0]-ukkt[0])<1e-9 )
# Value predicted by the linearization of the transition model:
#      xlin = xpred + Fx dx + Fu du = f(xguess,ugess) + Fx (xddp-xguess) + Fu (uddp-ugess).
xddplin1 = model.calc(model.createData(),xs[0],us[0])[0]      \
           + np.dot(problem.runningDatas[0].Fx,xddp[0]-xs[0]) \
           + np.dot(problem.runningDatas[0].Fu,uddp[0]-us[0])
assert( norm(xddplin1-xkkt[1])<1e-9 )

# --- TEST DDP VS KKT NLP in T time---
'''
This test computes the KKT solution on one step, and compare it with the DDP solution
obtained from the linearized dynamics about the current guess xs,us.
'''
model = ActionModelUnicycle()
nx = model.nx
nu = model.nu
T = 20

problem = ShootingProblem(model.State.zero()+2, [ model ]*T, model)
xs = [ np.random.rand(nx)   for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.random.rand(nu)   for m in problem.runningModels ]

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

kkt.setCandidate(xs,us)
dxkkt,dukkt,lkkt = kkt.computeDirection()
xkkt = [ x+dx for x,dx in zip(xs,dxkkt) ]
ukkt = [ u+du for u,du in zip(us,dukkt) ]

ddp.setCandidate(xs,us)
ddp.computeDirection()
xddp,uddp,costddp = ddp.forwardPass(stepLength=1)

assert( norm(problem.initialState[0]-xkkt[0])<1e-9 )
assert( norm(problem.initialState[0]-xddp[0])<1e-9 )
assert( norm(uddp[0]-ukkt[0])<1e-9 )

# Forward pass with linear dynamics.
ulin = [ np.nan ]*T
xlin = xddp[:1] + [ np.nan ]*T
for t in range(T):
    # Value predicted by the linearization of the transition model:
    #      xlin = xpred + Fx dx + Fu du = f(xguess,ugess) + Fx (xddp-xguess) + Fu (uddp-ugess).
    ulin[t] = us[t] - ddp.k[t] - np.dot(ddp.K[t],model.State.diff(xs[t],xlin[t]))
    xlin[t+1] = model.calc(model.createData(),xs[t],us[t])[0]    \
         + np.dot(problem.runningDatas[t].Fx,xlin[t]-xs[t])     \
         + np.dot(problem.runningDatas[t].Fu,ulin[t]-us[t])
    assert(norm(ulin[t]-ukkt[t])<1e-9)
    assert(norm(xlin[t+1]-xkkt[t+1])<1e-9)


# --- DDP NLP solver ---

np .random.seed     (220)
model = ActionModelLQR(1,1); model.setUpRandom();
nx = model.nx
nu = model.nu
T = 1

problem = ShootingProblem(model.State.zero()+2, [ model ]*T, model)
xs = [ m.State.rand()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.random.rand(nu)   for m in problem.runningModels ]
x0ref = problem.initialState

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)


xkkt,ukkt,donekkt=kkt.solve(maxiter=2,init_xs=xs,init_us=us)
xddp,uddp,doneddp=ddp.solve(maxiter=2,init_xs=xs,init_us=us,regInit=0)
assert(donekkt)
assert(doneddp)
assert(norm(xkkt[0]-problem.initialState)<1e-9)
assert(norm(xddp[0]-problem.initialState)<1e-9)
for t in range(problem.T):
    assert(norm(ukkt[t]-uddp[t])<1e-9)
    assert(norm(xkkt[t+1]-xddp[t+1])<1e-9)
    
# --- DDP VERSUS KKT : integrative test ---
np .random.seed     (220)
model = ActionModelUnicycle()
nx = model.nx
nu = model.nu
T = 20

problem = ShootingProblem(model.State.zero()+2, [ model ]*T, model)
xs = [ m.State.rand()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.random.rand(nu)   for m in problem.runningModels ]
x0ref = problem.initialState

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

ddp.th_stop = 1e-18
kkt.th_stop = 1e-18

xkkt,ukkt,donekkt=kkt.solve(maxiter=200,init_xs=xs,init_us=us)
xddp,uddp,doneddp=ddp.solve(maxiter=200,init_xs=xs,init_us=us,regInit=0)
assert(donekkt)
assert(doneddp)
assert(norm(xkkt[0]-problem.initialState)<1e-9)
assert(norm(xddp[0]-problem.initialState)<1e-9)
for t in range(problem.T):
    assert(norm(ukkt[t]-uddp[t])<1e-6)
    assert(norm(xkkt[t+1]-xddp[t+1])<1e-6)

# --- Test with manifold dynamics
model = ActionModelUnicycleVar()
data  = model.createData()

nx  = model.nx
ndx = model.ndx
nu  = model.nu
T = 10

x0ref = np.array([ -1,-1,1,0 ])
problem = ShootingProblem(x0ref, [ model ]*T, model)
xs = [ m.State.rand()   for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.random.rand(nu)     for m in problem.runningModels ]

kkt = SolverKKT(problem)
kkt.setCandidate(xs,us)
kkt.computeDirection()

xTOx3 = lambda x : model.State.diff     (model.State.zero(),x)
x3TOx = lambda x3: model.State.integrate(model.State.zero(),x3)

xs = problem.rollout(us)

model3 = ActionModelUnicycle()
data3  = model3.createData()

x0ref3 = xTOx3(x0ref)
problem3 = ShootingProblem(x0ref3, [ model3 ]*T, model3)
xs3 = [ xTOx3(x) for x in xs ]
us3 = [ u.copy() for u in us ]

kkt3 = SolverKKT(problem3)
kkt3.setCandidate(xs3,us3)
kkt3.computeDirection()

kkt.setCandidate(xs,us)
kkt.computeDirection()

assert( norm(kkt3.primal[:3]-kkt.primal[:3]) <1e-9 )

kkt.th_stop = 1e-18
kkt3.th_stop = 1e-18
x3,u3,d3 = kkt3.solve(maxiter=100,init_xs=xs3,init_us=us3,verbose=False)
x4,u4,d4 = kkt .solve(maxiter=100,init_xs=xs ,init_us=us,verbose=False)
assert( d3 and d4 )

assert( norm( X.diff(x4[0],x0ref) ) < 1e-9 )
assert( norm( x3[0]-x0ref3 ) < 1e-9 )
for t in range(T):
    assert( norm( u4[t]-u3[t] ) < 1e-9 )
    assert( norm( x4[t+1]-x3TOx(x3[t+1]) ) < 1e-9 )
assert( norm(kkt.primal - kkt3.primal) <1e-9 )
# Duals are not equals as the jacobians are not the same.


del problem
# -------------------------------------------------------------------
# --- REG -----------------------------------------------------------
# -------------------------------------------------------------------
model = ActionModelLQR(1,1); model.setUpRandom()
action = model.createData()
nx = model.nx
nu = model.nu
T = 1

problem = ShootingProblem(model.State.zero()+1, [model]*T,model)
xs = [ m.State.rand()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.random.rand(nu)   for m in problem.runningModels ]
x0ref = problem.initialState

kkt = SolverKKT(problem)
xkkt,ukkt,dkkt = kkt.solve(maxiter=2)
assert(dkkt)

kkt.setCandidate(xs,us)
dxs,dus,ls = kkt.computeDirection()

xs = [ m.State.zero()   for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.zeros(nu)     for m in problem.runningModels ]
kkt.setCandidate(xs,us)
kkt.x_reg = .1
kkt.u_reg = 1
dxs_reg,dus_reg,ls_reg = kkt.computeDirection()

import copy
modeldmp = copy.copy(model)
modeldmp.L[range(nx),      range(nx)      ] += kkt.x_reg
modeldmp.L[range(nx,nx+nu),range(nx,nx+nu)] += kkt.u_reg
problemdmp = ShootingProblem(model.State.zero()+1, [modeldmp]*T,modeldmp)
kktdmp = SolverKKT(problem)

kktdmp.setCandidate(kkt.xs,kkt.us)
dxs_dmp,dus_dmp,ls_dmp = kktdmp.computeDirection()
for t in range(T):
    assert(norm( dus_dmp[t  ]-dus_reg[t  ] ) <1e-9 )
    assert(norm( dxs_dmp[t+1]-dxs_reg[t+1] ) <1e-9 )

    
# --- REG UNICYCLE ---
model = ActionModelUnicycleVar()
data  = model.createData()

nx  = model.nx
ndx = model.ndx
nu  = model.nu
T = 10

x0ref = np.array([ -1,-1,1,0 ])
problem = ShootingProblem(x0ref, [ model ]*T, model)
xs = [ m.State.zero()   for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.zeros(nu)     for m in problem.runningModels ]

kkt = SolverKKT(problem)
kkt.setCandidate(xs,us)
kkt.x_reg = 1
kkt.u_reg = 1
model.costWeights[0] = 0
model.costWeights[1] = 0
dxs,dus,ls = kkt.computeDirection()

modeldmp = copy.copy(model)
problemdmp = ShootingProblem(x0ref, [ modeldmp ]*T, modeldmp)
kktdmp = SolverKKT(problem)
kktdmp.setCandidate(xs,us)
modeldmp.costWeights[0] = kkt.x_reg 
modeldmp.costWeights[1] = kkt.u_reg 
dxs_dmp,dus_dmp,ls_dmp = kktdmp.computeDirection()

for t in range(T):
    assert(norm( dus_dmp[t  ]-dus[t  ] ) <1e-9 )
    assert(norm( dxs_dmp[t+1]-dxs[t+1] ) <1e-9 )

# --- DDP --- 
model = ActionModelLQR(3,3); model.setUpRandom()
action = model.createData()
nx = model.nx
nu = model.nu
T = 5

problem = ShootingProblem(model.State.zero()+1, [model]*T,model)
xs = [ m.State.rand()       for m in problem.runningModels + [problem.terminalModel] ]
us = [ np.random.rand(nu)   for m in problem.runningModels ]
x0ref = problem.initialState

kkt = SolverKKT(problem)
ddp = SolverDDP(problem)

kkt.setCandidate(xs,us)
ddp.setCandidate(xs,us)
def deltaddp(solv):
    solv.computeDirection()
    xs_d,us_d,cost_d = solv.forwardPass(stepLength=1)
    dxs_d = [ m.State.diff(x0,x) for m,x0,x in zip(solv.models(),xs,xs_d) ]
    dus_d = [ u-u0 for u0,u in zip(us,us_d) ]
    return dxs_d,dus_d

dxs_k,dus_k,ls_k = kkt.computeDirection()
dxs_d,dus_d = deltaddp(ddp)
for t in range(T):
    assert(norm( dus_d[t  ]-dus_k[t  ] ) <1e-9 )
    assert(norm( dxs_d[t+1]-dxs_k[t+1] ) <1e-9 )

kkt.u_reg = 1
ddp.u_reg = 1

dxs_k,dus_k,ls_k = kkt.computeDirection()
dxs_d,dus_d = deltaddp(ddp)
for t in range(T):
    assert(norm( dus_d[t  ]-dus_k[t  ] ) <1e-9 )
    assert(norm( dxs_d[t+1]-dxs_k[t+1] ) <1e-9 )

kkt.x_reg = 1
ddp.x_reg = 1

dxs_k,dus_k,ls_k = kkt.computeDirection()
dxs_d,dus_d = deltaddp(ddp)
for t in range(T):
    assert(norm( dus_d[t  ]-dus_k[t  ] ) <1e-9 )
    assert(norm( dxs_d[t+1]-dxs_k[t+1] ) <1e-9 )

# --- REG INTEGRATIVE TEST ---
