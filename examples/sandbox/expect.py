from crocoddyl import *
import numpy as np
import sys
from numpy.linalg import norm,svd,pinv,inv
from numpy import dot
from crocoddyl.fddp import SolverFDDP

np.random.seed(0)

NX,NU = 2,1
T = 10
models = [ ActionModelLQR(NX,NU,driftFree= False) for t in range(T+1) ]

#models[1].Lx[:] = 0
#models[0].Lx[:] = 0
#models[0].Fx[:,:] = 0
#models[0].Fu[:,:] = 0

x0 = np.array([ 1. ]*NX)
problem = ShootingProblem(x0, models[:-1],models[-1])

ddp = SolverFDDP(problem)
ddp.regMin=0
kkt = SolverKKT(problem)

[xskkt,uskkt,donekkt] = kkt.solve(maxiter=1)
#[xsddp,usddp,doneddp] = ddp.solve(maxiter=1,regInit=0)

ddp.x_reg = 0
ddp.u_reg = 0
ddp.setCandidate()
ddp.computeDirection()
ddp.tryStep(1)

for t in range(T):
    assert( np.allclose(ddp.us_try[t],kkt.us[t]) )
    assert( np.allclose(ddp.xs_try[t+1],kkt.xs[t+1]) )


#assert(np.isclose(-dot(ddp.Vx[0],x0)+ddp.expectedImprovement()[0],kkt.expectedImprovement()[0]))

datas = ddp.datas()
Fu  = [ np.matrix(d.Fu)  for d in datas ]
Fx  = [ np.matrix(d.Fx)  for d in datas ]
Lu  = [ a2m(d.Lu)  for d in datas ]
Lx  = [ a2m(d.Lx)  for d in datas ]
Lxx = [ np.matrix(d.Lxx) for d in datas ]
Vx =  [ a2m(v) for v in ddp.Vx ]
Qx =  [ a2m(q) for q in ddp.Qx ]
Qu =  [ a2m(q) for q in ddp.Qu ]
k  =  [ a2m(k) for k in ddp.k  ]
K  =  [ np.matrix(K) for K in ddp.K ]
Vxx  =  [ np.matrix(Vxx) for Vxx in ddp.Vxx ]
f  =  [ a2m(f) for f in ddp.gaps ]

x = [ a2m(x) for x in kkt.xs ]
u = [ a2m(u) for u in kkt.us ]

FuK = [ -_Fu*_K for _Fu,_K in zip(Fu[:-1],K) ]
KtFut = [ -_K.T*_Fu.T for _Fu,_K in zip(Fu[:-1],K) ]
KtLu = [ -_K.T*_lu for _lu,_K in zip(Lu[:-1],K) ]

    


if T==1:
    assert(np.isclose( Lx[0].T*x[0]+Lu[0].T*u[0]+Lx[1].T*x[1], -kkt.expectedImprovement()[0] ) )
    assert(np.isclose( -Lx[0].T*f[0]+Lu[0].T*(-k[0]+K[0]*f[0])+Lx[1].T*x[1], -kkt.expectedImprovement()[0] ) )
    assert(np.isclose( (Lx[0]-K[0].T*Lu[0]).T*-f[0]+Lu[0].T*-k[0] \
                       +Lx[1].T*(Fu[0]*-k[0] + (Fx[0]-Fu[0]*K[0])*x[0]-f[1]), -kkt.expectedImprovement()[0] ) )
    assert(np.isclose( (Lx[0]+KtLu[0]+(Fx[0].T+KtFut[0])*Lx[1]).T*-f[0] \
                       +(Lu[0] +Fu[0].T*Lx[1]).T*-k[0] \
                       +Lx[1].T*-f[1], -kkt.expectedImprovement()[0] ) )
    assert( np.isclose( dot((KtLu[0]+Lx[0]+dot(KtFut[0]+Fx[0].T,Lx[1])).T,-f[0]) \
                       + dot((Lu[0]+dot(Fu[0].T,Lx[1])).T,-k[0].T) \
                       + dot(Lx[1].T,-f[1]), 
                       -kkt.expectedImprovement()[0] ))

alpha = Lx[-1]
expect = alpha.T*f[-1]
for t in range(T-1,-1,-1):
    expect += (Lu[t] + Fu[t].T*alpha).T*k[t]
    alpha = -K[t].T*Lu[t] + Lx[t] + (-K[t].T*Fu[t].T+Fx[t].T)*alpha
    expect += alpha.T*f[t]

assert( np.isclose( expect,kkt.expectedImprovement()[0]) )
