from crocoddyl import IntegratedActionModelEuler,StateVector
import numpy as np

class DifferentialActionModelCartPole:
    def __init__(self):
        self.State = StateVector(4)
        self.nq,self.nv = 2,2
        self.nx = 4
        self.ndx = 4
        self.nout = 2
        self.nu = 1
        self.ncost = 6
        self.unone = np.zeros(self.nu)

        self.m1 = 1.
        self.m2 = .1
        self.l  = .5
        self.g  = 9.81
        self.costWeights = [ 1., 1., 0.1, 0.001, .001, 1. ]  # sin, cos, x, xdot, thdot, f

    def createData(self): return DifferentialActionDataCartPole(self)
    def calc(model,data,x,u=None):
        if u is None: u=model.unone

        nx,nu,nq,nv,nout = model.nx,model.nu,model.nq,model.nv,model.nout
        x,th,xdot,thdot = x
        f, = u

        m1,m2,l,g = model.m1,model.m2,model.l,model.g
        s,c = np.sin(th),np.cos(th)
        m = m1+m2
        
        mu = m1+m2*s**2
        xddot  = (f     + m2*c*s*g - m2*l*s*thdot**2 )/mu
        thddot = (c*f/l + m*g*s/l  - m2*c*s*thdot**2 )/mu

        data.xout[:] = [ xddot,thddot ]
        data.costResiduals[:] =  [ s,c, x, xdot, thdot, f ]
        data.costResiduals[:] *= model.costWeights
        data.cost = .5*sum( data.costResiduals**2 )
        
        return data.xout,data.cost

    def calcDiff(model,data,x,u=None,recalc=True):
        pass
    
class DifferentialActionDataCartPole:
    def __init__(self,model):
        self.cost = np.nan
        self.xout = np.zeros(model.nout)

        nx,nu,ndx,nq,nv,nout = model.nx,model.nu,model.State.ndx,model.nq,model.nv,model.nout
        self.costResiduals = np.zeros([ model.ncost ])
        self.g   = np.zeros([ ndx+nu ])
        self.L   = np.zeros([ ndx+nu,ndx+nu ])
        self.F   = np.zeros([ nout,ndx+nu ])

        self.Lx  = self.g[:ndx]
        self.Lu  = self.g[ndx:]
        self.Lxx = self.L[:ndx,:ndx]
        self.Lxu = self.L[:ndx,ndx:]
        self.Luu = self.L[ndx:,ndx:]
        self.Fx  = self.F[:,:ndx]
        self.Fu  = self.F[:,ndx:]


cartpole = model = DifferentialActionModelCartPole()
data  = model.createData()

x = model.State.rand()
u = np.zeros(1)
model.calc(data,x,u)


from crocoddyl import DifferentialActionModelNumDiff

model = DifferentialActionModelNumDiff(model,withGaussApprox=True)
data  = model.createData()

model.calcDiff(data,x,u)


from crocoddyl import IntegratedActionModelEuler

model = IntegratedActionModelEuler(model)
data  = model.createData()

model.timeStep = 5e-2

model.calc(data,x,u)
model.calcDiff(data,x,u)

termCartpole = DifferentialActionModelCartPole()
termModel = IntegratedActionModelEuler(DifferentialActionModelNumDiff(termCartpole,withGaussApprox=True))
cartpole.costWeights[1] = 10
termCartpole.costWeights[0] = 10000
termCartpole.costWeights[1] = 10000
termCartpole.costWeights[3] = 1
termCartpole.costWeights[4] = 10


from crocoddyl import ShootingProblem,SolverDDP,CallbackDDPVerbose
x0 = np.array([ 0., 3.15, 0., 0. ])
T  = 50
problem = ShootingProblem(x0, [ model ]*T, termModel)



ddp = SolverDDP(problem)
#ddp.callback = [ CallbackDDPVerbose() ]
xs,us,done = ddp.solve()

for i in range(1,5):
    termCartpole.costWeights[4] = 100*i
    termCartpole.costWeights[3] = 10*i
    xs,us,done = ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=10)
    print xs[-1]

import cartpole_utils
