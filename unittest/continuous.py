import refact
import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,norm,pinv
from numpy import dot,asarray
from scipy.linalg import block_diag

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T

path = '/home/nmansard/src/cddp/examples/'

urdf = path + 'talos_data/robots/talos_left_arm.urdf'
robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(urdf, [path] \
#                                                           ,pinocchio.JointModelFreeFlyer() \
)

#urdf = path + 'hyq_description/robots/hyq_no_sensors.urdf'
#robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(urdf, [path], pinocchio.JointModelFreeFlyer())
qmin = robot.model.lowerPositionLimit; qmin[:7]=-1; robot.model.lowerPositionLimit = qmin
qmax = robot.model.upperPositionLimit; qmax[:7]= 1; robot.model.upperPositionLimit = qmax

rmodel = robot.model
rdata = rmodel.createData()

q0 = pinocchio.randomConfiguration(rmodel)
dq = rand(rmodel.nv)*2-1
q  = pinocchio.integrate(rmodel,q0,dq)
diff= pinocchio.difference(rmodel,q0,q)
assert(norm(dq-diff)<1e-9)

q0 = pinocchio.randomConfiguration(rmodel)
v  = rand(rmodel.nv)*2-1
q  = pinocchio.integrate(rmodel,q0,v)
Q = lambda _v: pinocchio.integrate(rmodel,q0,_v)
V = lambda _q: pinocchio.difference(rmodel,q0,_q)
assert(norm(Q(V(q))-q)<1e-9)
assert(norm(V(Q(v))-v)<1e-9)

class StatePinocchio:
    def __init__(self,pinocchioModel):
        self.model = pinocchioModel
        self.nx = self.model.nq + self.model.nv
        self.ndx = 2*self.model.nv
    def zero(self):
        q = self.model.neutralConfiguration
        v = np.zeros(self.model.nv)
        return np.concatenate([q.flat,v])
    def rand(self):
        q = pinocchio.randomConfiguration(self.model)
        v = np.random.rand(self.model.nv)*2-1
        return np.concatenate([q.flat,v])
    def diff(self,x0,x1):
        nq,nv,nx,ndx = self.model.nq,self.model.nv,self.nx,self.ndx
        assert( x0.shape == ( nx, ) and x1.shape == ( nx, ))
        q0 = x0[:nq]; q1 = x1[:nq]; v0 = x0[-nv:]; v1 = x1[-nv:]
        dq = pinocchio.difference(self.model,a2m(q0),a2m(q1))
        return np.concatenate([dq.flat,v1-v0])
    def integrate(self,x,dx):
        nq,nv,nx,ndx = self.model.nq,self.model.nv,self.nx,self.ndx
        assert( x.shape == ( nx, ) and dx.shape == ( ndx, ))
        q = x[:nq]; v = x[-nv:]; dq = dx[:nv]; dv = dx[-nv:]
        qn = pinocchio.integrate(self.model,a2m(q),a2m(dq))
        return np.concatenate([ qn.flat, v+dv] )
    def Jdiff(self,x1,x2,firstsecond='both'):
        assert(firstsecond in ['first', 'second', 'both' ])
        if firstsecond == 'both': return [ self.Jdiff(x1,x2,'first'),
                                           self.Jdiff(x1,x2,'second') ]
        if firstsecond == 'second':
            dx = self.diff(x1,x2)
            q  = a2m( x1[:self.model.nq])
            dq = a2m( dx[:self.model.nv])
            Jdq = pinocchio.dIntegrate(self.model,q,dq)[1]
            return  block_diag( asarray(inv(Jdq)), np.eye(self.model.nv) )
        else:
            dx = self.diff(x2,x1)
            q  = a2m( x2[:self.model.nq])
            dq = a2m( dx[:self.model.nv])
            Jdq = pinocchio.dIntegrate(self.model,q,dq)[1]
            return -block_diag( asarray(inv(Jdq)), np.eye(self.model.nv) )
        
    def Jintegrate(self,x,dx,firstsecond='both'):
        assert(firstsecond in ['first', 'second', 'both' ])
        assert(x.shape == ( self.nx, ) and dx.shape == (self.ndx,) )
        if firstsecond == 'both': return [ self.Jintegrate(x,dx,'first'),
                                           self.Jintegrate(x,dx,'second') ]
        q  = a2m( x[:self.model.nq])
        dq = a2m(dx[:self.model.nv])
        Jq,Jdq = pinocchio.dIntegrate(self.model,q,dq)
        if firstsecond=='first':
            # Derivative wrt x
            return block_diag( asarray(Jq), np.eye(self.model.nv) )
        else:
            # Derivative wrt v
            return block_diag( asarray(Jdq), np.eye(self.model.nv) )
    
# Check integrate is reciprocal of diff.
X = StatePinocchio(rmodel)
x0 = X.zero()
x0 = X.rand()
dx = np.random.rand(rmodel.nv*2)
x1 = X.integrate(x0,dx)
assert(norm(X.diff(x0,x1)-dx)<1e-9)

# Check integrate and dIntegrate of pinocchio
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)*2-1
J1,J2 = pinocchio.dIntegrate(rmodel,q,v)
h = 1e-12
dv = zero(rmodel.nv)
for k in range(rmodel.nv):
    dv[k] = 1
    i0 = pinocchio.integrate(rmodel,q,v)
    i1 = pinocchio.integrate(rmodel,q,v+dv*h)
    j = pinocchio.difference(rmodel,i0,i1)/h
    assert(norm(j-J2*dv)<1e-3)
    i0 = pinocchio.integrate(rmodel,q,v)
    i1 = pinocchio.integrate(rmodel,pinocchio.integrate(rmodel,q,dv*h),v)
    j = pinocchio.difference(rmodel,i0,i1)/h
    assert(norm(j-J1*dv)<1e-3)

# Check safe call of X.Jintegrate
x = X.rand()
Jx,Jdx = X.Jintegrate(x,dx)

# Check .XJintegrate versus numdiff.
from refact import StateNumDiff
Xnum = StateNumDiff(X)
Xnum.disturbance=1e-12
x = X.rand()
dx = np.random.rand(X.ndx)
Jx,Jdx = X.Jintegrate(x,dx)
Jx_num,Jdx_num = Xnum.Jintegrate(x,dx)
assert(norm(Jx -Jx_num )<1e-2) # Very low num precision ... why? 
assert(norm(Jdx-Jdx_num)<1e-2) # ... problem in pinocchio-to-numpy conversion?

# Check safe call of Xnum.Jdiff
x1 = X.rand()
x2 = X.rand()
J1_num,J2_num = Xnum.Jdiff(x1,x2)

# Build X and DX maps to validate Jdiff is inv of Jintegrate.
x0 = X.rand()
fX = lambda _dx: X.integrate(x0,_dx)
fDX = lambda _x: X.diff(x0,_x)
x = X.rand()
dx = np.random.rand(X.ndx)*2-1
assert(norm(X.diff(fX(fDX(x)),x))<1e-9) 
assert(norm(fDX(fX(dx))-dx)<1e-9)

# Validate that Jdiff and Jintegrate are inverses.
#x = X.rand()
#dx = X.diff(x0,x)
#assert(norm(x-X.integrate(x0,dx))<1e-9 or abs(norm(x-X.integrate(x0,dx))-2)<1e-9)
dx = np.random.rand(X.ndx)*2-1
x = X.integrate(x0,dx)
assert(norm( dx-X.diff(x0,x) )<1e-9)
eps = np.random.rand(X.ndx)
eps*=0; eps[0]=1
J1_num,J2_num = Xnum.Jdiff(x0,x)
Jx,Jdx = X.Jintegrate(x0,dx)
dX_dDX = Jdx
dDX_dX = J2_num
# dX_dDX*eps =?= diff(X(dx),X(dx+eps))
assert(norm(np.dot(dX_dDX,eps)-X.diff(fX(dx),fX(dx+eps*h))/h)<1e-3)
assert(norm(np.dot(dDX_dX,eps)-(-fDX(x)+fDX(X.integrate(x,eps*h)))/h)<1e-3)
assert(norm( dX_dDX-inv(dDX_dX) ) <1e-2 )
assert(norm( dDX_dX-inv(dX_dDX) ) <1e-2 )

del(dx)
x1 = X.rand()
x1 = X.rand()
x2 = X.rand()

J1,J2 = X.Jdiff(x1,x2)
J1_num,J2_num = Xnum.Jdiff(x1,x2)
assert(norm(J2 -J2_num )<1e-2)
assert(norm(J1 -J1_num )<1e-2)


class DifferentialActionModel:
    def __init__(self,pinocchioModel):
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.nq,self.nv = self.pinocchio.nq, self.pinocchio.nv
        self.nx = self.State.nx
        self.ndx = self.State.ndx
        self.nout = self.nv
        self.nu = self.nv
        self.ncost = self.State.ndx + 3 + self.nu
        self.frameIdx = self.pinocchio.getFrameId('gripper_left_fingertip_2_link')
        self.pref = np.array([.5,.4,.3])
        self.qref = np.zeros(self.nq) + .3
        if self.pinocchio.joints[1].shortname()=='JointModelFreeFlyer':
            self.qref[6]+=1
            self.qref[3:7] /= norm(self.qref[3:7])
        self.vref = np.zeros(self.nv)
        self.xref = np.concatenate([ self.qref,self.vref ])
        self.costWeights = [1,.1,.001]
        self.unone = np.zeros(self.nu)
    def createData(self): return DifferentialActionData(self)
    
    def calc(model,data,x,u=None):
        if u is None: u=model.unone
        nx,nu,nq,nv,nout = model.nx,model.nu,model.nq,model.nv,model.nout
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(u)
        data.xout[:] = pinocchio.aba(model.pinocchio,data.pinocchio,q,v,tauq).flat
        pinocchio.forwardKinematics(model.pinocchio,data.pinocchio,q,v)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        data.costResiduals[:3] = model.costWeights[0]*(m2a(data.pinocchio.oMf[model.frameIdx].translation) - model.pref)
        data.costResiduals[3:3+model.ndx] = model.costWeights[1]*model.State.diff(model.xref,x)
        data.costResiduals[3+model.ndx:] = model.costWeights[2]*u
        data.cost = .5*norm(data.costResiduals)
        return data.xout,data.cost

    def calcDiff(model,data,x,u=None):
        if u is None: u=model.unone
        xout,cost = model.calc(data,x,u)
        nx,ndx,nu,nq,nv,nout = model.nx,model.State.ndx,model.nu,model.nq,model.nv,model.nout
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(u)
        pinocchio.computeABADerivatives(model.pinocchio,data.pinocchio,q,v,tauq)
        data.Fx[:,:nv] = data.pinocchio.ddq_dq
        data.Fx[:,nv:] = data.pinocchio.ddq_dv
        data.Fu[:,:]   = pinocchio.computeMinverse(model.pinocchio,data.pinocchio,q)
        
        R = data.pinocchio.oMf[model.frameIdx].rotation
        pinocchio.computeJointJacobians(model.pinocchio,data.pinocchio,q)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        J = pinocchio.getFrameJacobian(model.pinocchio,data.pinocchio,model.frameIdx,
                                       pinocchio.ReferenceFrame.LOCAL)
        data.Rx[:3,:nv] = model.costWeights[0]*R*J[:3,:]
        data.Rx[3:3+ndx,:ndx] = model.costWeights[1]*model.State.Jdiff(model.xref,x,'second')
        data.Ru[3+ndx:,:] = model.costWeights[2]*eye(nu)
        data.L[:,:] = np.dot(data.R.T,data.R)
        data.grad[:] = np.dot(data.R.T,data.costResiduals)
            
class DifferentialActionData:
    def __init__(self,model):
        self.pinocchio = model.pinocchio.createData()
        self.cost = np.nan
        self.costResiduals = np.zeros(model.ncost)
        self.xout = np.zeros(model.nout)
        nx,nu,ndx,nq,nv,nout = model.nx,model.nu,model.State.ndx,model.nq,model.nv,model.nout
        self.F = np.zeros([ nout,ndx+nu ])
        self.Fx = self.F[:,:ndx]
        self.Fu = self.F[:,-nu:]
        self. L = np.zeros([ ndx+nu, ndx+nu ])
        self.Lxx = self.L[:ndx,:ndx]
        self.Lxu = self.L[:ndx,ndx:]
        self.Lux = self.L[ndx:,:ndx]
        self.Luu = self.L[ndx:,ndx:]
        self.grad = np.zeros( ndx+nu )
        self.Lx = self.grad[:nx]
        self.Lu = self.grad[-nu:]
        self.R = np.zeros([ model.ncost,ndx+nu ])
        self.Rx = self.R[:,:ndx]
        self.Ru = self.R[:,-nu:]
        

model = DifferentialActionModel(rmodel)
data = model.createData()
q = m2a(pinocchio.randomConfiguration(rmodel))
v = np.random.rand(rmodel.nv)*2-1
x = np.concatenate([q,v])
u = np.random.rand(rmodel.nv)*2-1

a,l = model.calc(data,x,u)
model.calcDiff(data,x,u)

    
class DifferentialActionModelNumDiff:
    def __init__(self,model,withGaussApprox=False):
        self.model0 = model
        self.nx = model.nx
        self.ndx = model.ndx
        self.nout = model.nout
        self.nu = model.nu
        self.State = model.State
        self.disturbance = 1e-5
        self.ncost = model.ncost if 'ncost' in model.__dict__ else 1
        self.withGaussApprox = withGaussApprox
        assert( not self.withGaussApprox or self.ncost>1 )
        
    def createData(self):
        return DifferentialActionDataNumDiff(self)
    def calc(model,data,x,u): return model.model0.calc(data.data0,x,u)
    def calcDiff(model,data,x,u):
        xn0,c0 = model.calc(data,x,u)
        h = model.disturbance
        dist = lambda i,n,h: np.array([ h if ii==i else 0 for ii in range(n) ])
        Xint  = lambda x,dx: model.State.integrate(x,dx)
        for ix in range(model.ndx):
            xn,c = model.model0.calc(data.datax[ix],Xint(x,dist(ix,model.ndx,h)),u)
            data.Fx[:,ix] = (xn-xn0)/h
            data.Lx[  ix] = (c-c0)/h
            if model.ncost>1: data.Rx[:,ix] = (data.datax[ix].costResiduals-data.data0.costResiduals)/h
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
        nx,ndx,nu,ncost = model.nx,model.ndx,model.nu,model.ncost
        self.data0 = model.model0.createData()
        self.datax = [ model.model0.createData() for i in range(model.ndx) ]
        self.datau = [ model.model0.createData() for i in range(model.nu ) ]
        self.Lx = np.zeros([ model.ndx ])
        self.Lu = np.zeros([ model.nu ])
        self.Fx = np.zeros([ model.nout,model.ndx ])
        self.Fu = np.zeros([ model.nout,model.nu  ])
        if model.ncost >1 :
            self.Rx = np.zeros([model.ncost,model.ndx])
            self.Ru = np.zeros([model.ncost,model.nu ])
        if model.withGaussApprox:
            self. L = np.zeros([ ndx+nu, ndx+nu ])
            self.Lxx = self.L[:ndx,:ndx]
            self.Lxu = self.L[:ndx,ndx:]
            self.Lux = self.L[ndx:,:ndx]
            self.Luu = self.L[ndx:,ndx:]

mnum = DifferentialActionModelNumDiff(model,withGaussApprox=True)
dnum = mnum.createData()

model.calcDiff(data,x,u)
mnum.calcDiff(dnum,x,u)
thr = 1e-2 
assert( norm(data.Fx-dnum.Fx) < thr )
assert( norm(data.Fu-dnum.Fu) < thr )
assert( norm(data.Rx-dnum.Rx) < thr )
assert( norm(data.Ru-dnum.Ru) < thr )


# --- INTEGRATION ---
class IntegratedActionModelEuler:
    def __init__(self,diffModel):
        self.differential = diffModel
        self.State = self.differential.State
        self.nx    = self.differential.nx
        self.ndx   = self.differential.ndx
        self.nu    = self.differential.nu
        self.ncost = self.differential.ncost
        self.nq    = self.differential.nq
        self.nv    = self.differential.nv
        self.timeStep = 1e-3
    def createData(self): return IntegratedActionDataEuler(self)
    def calc(model,data,x,u):
        nx,ndx,nu,ncost,nq,nv,dt = model.nx,model.ndx,model.nu,model.ncost,model.nq,model.nv,model.timeStep
        acc,cost = model.differential.calc(data.differential,x,u)
        data.costResiduals[:] = data.differential.costResiduals[:]*dt
        data.cost = cost*dt
        data.xnext[nq:] = x[nq:] + acc*dt
        data.xnext[:nq] = pinocchio.integrate(model.differential.pinocchio,
                                              a2m(x[:nq]),a2m(data.xnext[nq:]*dt)).flat
        return data.xnext,data.cost
    def calcDiff(model,data,x,u):
        nx,ndx,nu,ncost,nq,nv,dt = model.nx,model.ndx,model.nu,model.ncost,model.nq,model.nv,model.timeStep
        
        
class IntegratedActionDataEuler:
    def __init__(self,model):
        nx,ndx,nu,ncost = model.nx,model.ndx,model.nu,model.ncost
        self.differential = model.differential.createData()

        self.g = np.zeros([ ndx+nu ])
        self.R = np.zeros([ ncost ,ndx+nu ])
        self.L = np.zeros([ ndx+nu,ndx+nu ])
        self.F = np.zeros([ ndx   ,ndx+nu ])
        self.xnext = np.zeros([ nx ])
        self.cost = np.nan
        self.costResiduals = np.zeros([ ncost ])
        
        self.Lxx = self.L[:ndx,:ndx]
        self.Lxu = self.L[:ndx,ndx:]
        self.Lux = self.L[ndx:,:ndx]
        self.Luu = self.L[ndx:,ndx:]
        self.Lx  = self.g[:ndx]
        self.Lu  = self.g[ndx:]
        self.Fx = self.F[:,:ndx]
        self.Fu = self.F[:,ndx:]
        self.Rx = self.R[:,:ndx]
        self.Ru = self.R[:,ndx:]

dmodel = DifferentialActionModel(rmodel)
ddata  = dmodel.createData()
model  = IntegratedActionModelEuler(dmodel)
data   = model.createData()

x = model.State.zero()
u = np.zeros( model.nu )
xn,c = model.calc(data,x,u)

stophere

# --- DDP FOR THE ARM ---
from refact import ShootingProblem,SolverKKT,SolverDDP
problem = ShootingProblem(model.State.zero()+1, [ model, model ], model)
kkt = SolverKKT(problem)


