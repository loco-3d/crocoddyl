import rospkg
import refact
import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,norm,pinv
from numpy import dot,asarray
from scipy.linalg import block_diag


m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
absmax = lambda A: np.max(abs(A))
absmin = lambda A: np.min(abs(A))

rospack = rospkg.RosPack()
#MODEL_PATH = rospack.get_path('talos_data')
MODEL_PATH = '/home/nmansard/src/cddp/examples'
MESH_DIR = MODEL_PATH
URDF_FILENAME = "talos_left_arm.urdf"
URDF_MODEL_PATH = MODEL_PATH + "/talos_data/robots/" + URDF_FILENAME
#URDF_MODEL_PATH = MODEL_PATH + "/robots/" + URDF_FILENAME

robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(URDF_MODEL_PATH, [MESH_DIR])

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


# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------

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

# --------------------------------------------------------------
        
class CostModelPosition(CostModelPinocchio):
    '''
    The class proposes a model of a cost function positioning (3d) 
    a frame of the robot. Paramterize it with the frame index frameIdx and
    the effector desired position ref.
    '''
    def __init__(self,pinocchioModel,frame,ref,nu=None):
        self.CostDataType = CostDataPosition
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=3,nu=nu)
        self.ref = ref
        self.frame = frame
    def calc(model,data,x,u):
        data.residuals = m2a(data.pinocchio.oMf[model.frame].translation) - model.ref
        data.cost = .5*sum(data.residuals**2)
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        ncost,nq,nv,nx,ndx,nu = model.ncost,model.nq,model.nv,model.nx,model.ndx,model.nu
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        R = data.pinocchio.oMf[model.frame].rotation
        J = R*pinocchio.getFrameJacobian(model.pinocchio,data.pinocchio,model.frame,
                                         pinocchio.ReferenceFrame.LOCAL)[:3,:]
        data.Rq[:,:nq] = J
        data.Lq[:]     = np.dot(J.T,data.residuals)
        data.Lqq[:,:]  = np.dot(J.T,J)
        return data.cost
    
class CostDataPosition(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.Lu = 0
        self.Lv = 0
        self.Lxu = 0
        self.Luu = 0
        self.Lvv = 0
        self.Ru = 0
        self.Rv = 0

        
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelPosition(rmodel,
                              rmodel.getFrameId('gripper_left_fingertip_2_link'),
                              np.array([.5,.4,.3]))

costData = costModel.createData(rdata)


pinocchio.forwardKinematics(rmodel,rdata,q,v)
pinocchio.computeJointJacobians(rmodel,rdata,q)
pinocchio.updateFramePlacements(rmodel,rdata)

costModel.calcDiff(costData,x,u)

costModelND = CostModelNumDiff(costModel,StatePinocchio(rmodel),withGaussApprox=True,
                               reevals = [ lambda m,d,x,u: pinocchio.forwardKinematics(m,d,a2m(x[:rmodel.nq]),a2m(x[rmodel.nq:])),
                                           lambda m,d,x,u: pinocchio.computeJointJacobians(m,d,a2m(x[:rmodel.nq])),
                                           lambda m,d,x,u: pinocchio.updateFramePlacements(m,d) ])
costDataND  = costModelND.createData(rdata)

costModelND.calcDiff(costDataND,x,u)

assert( absmax(costData.g-costDataND.g) < 1e-3 )
assert( absmax(costData.L-costDataND.L) < 1e-3 )

# --------------------------------------------------------------

class CostModelPosition6D(CostModelPinocchio):
    '''
    The class proposes a model of a cost function position and orientation (6d) 
    for a frame of the robot. Paramterize it with the frame index frameIdx and
    the effector desired pinocchio::SE3 ref.
    '''
    def __init__(self,pinocchioModel,frame,ref):
        self.CostDataType = CostDataPosition6D
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=6)
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
      
class CostDataPosition6D(CostDataPinocchio):
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

        
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelPosition6D(rmodel,
                                rmodel.getFrameId('gripper_left_fingertip_2_link'),
                                pinocchio.SE3(pinocchio.SE3.Random().rotation,
                                              np.matrix([.5,.4,.3]).T))

costData = costModel.createData(rdata)

pinocchio.forwardKinematics(rmodel,rdata,q,v)
pinocchio.computeJointJacobians(rmodel,rdata,q)
pinocchio.updateFramePlacements(rmodel,rdata)

costModel.calcDiff(costData,x,u)

costModelND = CostModelNumDiff(costModel,StatePinocchio(rmodel),withGaussApprox=True,
                               reevals = [ lambda m,d,x,u: pinocchio.forwardKinematics(m,d,a2m(x[:rmodel.nq]),a2m(x[rmodel.nq:])),
                                           lambda m,d,x,u: pinocchio.computeJointJacobians(m,d,a2m(x[:rmodel.nq])),
                                           lambda m,d,x,u: pinocchio.updateFramePlacements(m,d) ])
costDataND  = costModelND.createData(rdata)

costModelND.calcDiff(costDataND,x,u)

assert( absmax(costData.g-costDataND.g) < 1e-4 )
assert( absmax(costData.L-costDataND.L) < 1e-4 )

# --------------------------------------------------------------

class CostModelCoM(CostModelPinocchio):
    '''
    The class proposes a model of a cost function CoM. 
    Paramterize it with the desired CoM ref
    '''
    def __init__(self,pinocchioModel,ref):
        self.CostDataType = CostDataCoM
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=3)
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


q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelCoM(rmodel,
                         np.array([.5,.4,.3]))

costData = costModel.createData(rdata)


pinocchio.jacobianCenterOfMass(rmodel, rdata, q, False)

costModel.calcDiff(costData,x,u)

costModelND = CostModelNumDiff(costModel,StatePinocchio(rmodel),withGaussApprox=True,
                               reevals = [ lambda m,d,x,u: pinocchio.jacobianCenterOfMass(m,d,a2m(x[:rmodel.nq]),False)])
costDataND  = costModelND.createData(rdata)

costModelND.calcDiff(costDataND,x,u)

assert( absmax(costData.g-costDataND.g) < 1e-4 )
assert( absmax(costData.L-costDataND.L) < 1e-4 )

# --------------------------------------------------------------

class CostModelState(CostModelPinocchio):
    def __init__(self,pinocchioModel,State,ref,nu=None):
        self.CostDataType = CostDataState
        CostModelPinocchio.__init__(self,pinocchioModel,ncost=State.ndx,nu=nu)
        self.State = State
        self.ref = ref
    def calc(model,data,x,u):
        data.residuals[:] = model.State.diff(model.ref,x)
        data.cost = .5*sum(data.residuals**2)
        return data.cost
    def calcDiff(model,data,x,u,recalc=True):
        if recalc: model.calc(data,x,u)
        data.Rx[:,:] = model.State.Jdiff(model.ref,x,'second')
        data.Lx[:] = np.dot(data.Rx.T,data.residuals)
        data.Lxx[:,:] = np.dot(data.Rx.T,data.Rx)
        
class CostDataState(CostDataPinocchio):
    def __init__(self,model,pinocchioData):
        CostDataPinocchio.__init__(self,model,pinocchioData)
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0

X = StatePinocchio(rmodel)        
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelState(rmodel,X,X.rand())
costData = costModel.createData(rdata)
costModel.calcDiff(costData,x,u)

costModelND = CostModelNumDiff(costModel,X,withGaussApprox=True,
                               reevals = [])
costDataND  = costModelND.createData(rdata)
costModelND.calcDiff(costDataND,x,u)

assert( absmax(costData.g-costDataND.g) < 1e-3 )
assert( absmax(costData.L-costDataND.L) < 1e-3 )
    
# --------------------------------------------------------------

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
  
X = StatePinocchio(rmodel)
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelControl(rmodel)
costData = costModel.createData(rdata)
costModel.calcDiff(costData,x,u)

costModelND = CostModelNumDiff(costModel,StatePinocchio(rmodel),withGaussApprox=True)
costDataND  = costModelND.createData(rdata)
costModelND.calcDiff(costDataND,x,u)

assert( absmax(costData.g-costDataND.g) < 1e-3 )
assert( absmax(costData.L-costDataND.L) < 1e-3 )
    
# --------------------------------------------------------------
from collections import OrderedDict

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

X = StatePinocchio(rmodel)
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q,v]))
u = m2a(rand(rmodel.nv))

cost1 = CostModelPosition(rmodel,
                              rmodel.getFrameId('gripper_left_fingertip_2_link'),
                              np.array([.5,.4,.3]))
cost2 = CostModelState(rmodel,X,X.rand())
cost3 = CostModelControl(rmodel)

costModel = CostModelSum(rmodel)
costModel.addCost("pos",cost1,10)
costModel.addCost("regx",cost2,.1)
costModel.addCost("regu",cost3,.01)
costData = costModel.createData(rdata)

pinocchio.forwardKinematics(rmodel,rdata,q,v)
pinocchio.computeJointJacobians(rmodel,rdata,q)
pinocchio.updateFramePlacements(rmodel,rdata)
costModel.calcDiff(costData,x,u)

costModelND = CostModelNumDiff(costModel,StatePinocchio(rmodel),withGaussApprox=True,
                               reevals = [ lambda m,d,x,u: pinocchio.forwardKinematics(m,d,a2m(x[:rmodel.nq]),a2m(x[rmodel.nq:])),
                                           lambda m,d,x,u: pinocchio.computeJointJacobians(m,d,a2m(x[:rmodel.nq])),
                                           lambda m,d,x,u: pinocchio.updateFramePlacements(m,d) ])
costDataND  = costModelND.createData(rdata)
costModelND.calcDiff(costDataND,x,u)

assert( absmax(costData.g-costDataND.g) < 1e-3 )
assert( absmax(costData.L-costDataND.L) < 1e-3 )

# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------



class DifferentialActionModel:
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
    @property
    def ncost(self): return self.costs.ncost
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
        data.cost = model.costs.calc(data.costs,x,u)
        return data.xout,data.cost

    def calcDiff(model,data,x,u=None,recalc=True):
        if u is None: u=model.unone
        if recalc: xout,cost = model.calc(data,x,u)
        nx,ndx,nu,nq,nv,nout = model.nx,model.State.ndx,model.nu,model.nq,model.nv,model.nout
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(u)
        pinocchio.computeABADerivatives(model.pinocchio,data.pinocchio,q,v,tauq)
        data.Fx[:,:nv] = data.pinocchio.ddq_dq
        data.Fx[:,nv:] = data.pinocchio.ddq_dv
        data.Fu[:,:]   = pinocchio.computeMinverse(model.pinocchio,data.pinocchio,q)

        pinocchio.computeJointJacobians(model.pinocchio,data.pinocchio,q)
        pinocchio.updateFramePlacements(model.pinocchio,data.pinocchio)
        model.costs.calcDiff(data.costs,x,u,recalc=False)
        
        return data.xout,data.cost

class DifferentialActionData:
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

class DifferentialActionModelPositioning(DifferentialActionModel):
    def __init__(self,pinocchioModel,frameName='gripper_left_fingertip_2_link'):
        DifferentialActionModel.__init__(self,pinocchioModel)
        self.costs.addCost( name="pos", weight = 10,
                            cost = CostModelPosition(pinocchioModel,
                                                     pinocchioModel.getFrameId(frameName),
                                                     np.array([.5,.4,.3])))
        self.costs.addCost( name="regx", weight = 0.1,
                            cost = CostModelState(pinocchioModel,self.State,
                                                  self.State.zero()) )
        self.costs.addCost( name="regu", weight = 0.01,
                            cost = CostModelControl(pinocchioModel) )
        
q = m2a(pinocchio.randomConfiguration(rmodel))
v = np.random.rand(rmodel.nv)*2-1
x = np.concatenate([q,v])
u = np.random.rand(rmodel.nv)*2-1

model = DifferentialActionModelPositioning(rmodel)
data = model.createData()

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
        self.nq    = self.differential.nq
        self.nv    = self.differential.nv
        self.timeStep = 1e-3
    @property
    def ncost(self): return self.differential.ncost
    def createData(self): return IntegratedActionDataEuler(self)
    def calc(model,data,x,u=None):
        nx,ndx,nu,ncost,nq,nv,dt = model.nx,model.ndx,model.nu,model.ncost,model.nq,model.nv,model.timeStep
        acc,cost = model.differential.calc(data.differential,x,u)
        data.costResiduals[:] = data.differential.costResiduals[:]
        data.cost = cost
        # data.xnext[nq:] = x[nq:] + acc*dt
        # data.xnext[:nq] = pinocchio.integrate(model.differential.pinocchio,
        #                                       a2m(x[:nq]),a2m(data.xnext[nq:]*dt)).flat
        data.dx = np.concatenate([ x[nq:]*dt+acc*dt**2, acc*dt ])
        data.xnext[:] = model.differential.State.integrate(x,data.dx)

        return data.xnext,data.cost
    def calcDiff(model,data,x,u=None,recalc=True):
        nx,ndx,nu,ncost,nq,nv,dt = model.nx,model.ndx,model.nu,model.ncost,model.nq,model.nv,model.timeStep
        if recalc: model.calc(data,x,u)
        model.differential.calcDiff(data.differential,x,u,recalc=False)
        dxnext_dx,dxnext_ddx = model.State.Jintegrate(x,data.dx)
        da_dx,da_du = data.differential.Fx,data.differential.Fu
        ddx_dx = np.vstack([ da_dx*dt, da_dx ]); ddx_dx[range(nv),range(nv,2*nv)] += 1
        data.Fx[:,:] = dxnext_dx + dt*np.dot(dxnext_ddx,ddx_dx)
        ddx_du = np.vstack([ da_du*dt, da_du ])
        data.Fu[:,:] = dt*np.dot(dxnext_ddx,ddx_du)
        data.g[:] = data.differential.g
        data.L[:] = data.differential.L
        
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

dmodel = DifferentialActionModelPositioning(rmodel)
ddata  = dmodel.createData()
model  = IntegratedActionModelEuler(dmodel)
data   = model.createData()

x = model.State.zero()
u = np.zeros( model.nu )
xn,c = model.calc(data,x,u)

model.timeStep = 1
model.differential.costs
for k in model.differential.costs.costs.keys(): model.differential.costs[k].weight = 1

model.calcDiff(data,x,u)

from refact import ActionModelNumDiff
mnum = ActionModelNumDiff(model,withGaussApprox=True)
dnum = mnum.createData()

mnum.calcDiff(dnum,x,u)
assert( norm(data.Fx-dnum.Fx) < np.sqrt(mnum.disturbance)*10 )
assert( norm(data.Fu-dnum.Fu) < np.sqrt(mnum.disturbance)*10 )
assert( norm(data.Lx-dnum.Lx) < 10*np.sqrt(mnum.disturbance) )
assert( norm(data.Lu-dnum.Lu) < 10*np.sqrt(mnum.disturbance) )
assert( norm(dnum.Lxx-data.Lxx) < 10*np.sqrt(mnum.disturbance) )
assert( norm(dnum.Lxu-data.Lxu) < 10*np.sqrt(mnum.disturbance) )
assert( norm(dnum.Luu-data.Luu) < 10*mnum.disturbance )

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# --- DDP FOR THE ARM ---
dmodel = DifferentialActionModelPositioning(rmodel)
model  = IntegratedActionModelEuler(dmodel)

from refact import ShootingProblem,SolverKKT,SolverDDP
problem = ShootingProblem(model.State.zero()+1, [ model ], model)
kkt = SolverKKT(problem)
kkt.th_stop = 1e-18
xkkt,ukkt,dkkt = kkt.solve()

ddp = SolverDDP(problem)
ddp.th_stop = 1e-18
xddp,uddp,dddp = ddp.solve()

assert( norm(uddp[0]-ukkt[0]) < 1e-6 )


