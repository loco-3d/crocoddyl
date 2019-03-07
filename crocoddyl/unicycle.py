import numpy as np
from numpy import arctan2, cos, sin

from action import ActionDataLQR, ActionModelLQR
from state import StateAbstract, StateVector


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




class StateUnicycle(StateAbstract):
    def __init__(self):
        StateAbstract.__init__(self,4,3)
        pass

    def zero(self):   return np.array([0.,0.,1.,0.]) # a,b,c,s
    def rand(self):
        a,b,th = np.random.rand(3)
        return np.array([a,b,cos(th),sin(th)])
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
