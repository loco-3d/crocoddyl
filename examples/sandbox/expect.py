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
x0 = np.array([ 1. ]*NX)
problem = ShootingProblem(x0, models[:-1],models[-1])

#models[1].Lx[:] = 0
#models[0].Lx[:] = 0
#models[0].Fx[:,:] = 0
#models[0].Fu[:,:] = 0
#x0[:] =0
#models[0].f0[:] = 0

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
Quu = [ np.matrix(Q) for Q in ddp.Quu ]
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

alpha = [np.nan]*T + [Lx[-1]]
expect = alpha[T].T*f[-1]
for t in range(T-1,-1,-1):
    expect += (Lu[t] + Fu[t].T*alpha[t+1]).T*k[t]
    alpha[t] = -K[t].T*Lu[t] + Lx[t] + (-K[t].T*Fu[t].T+Fx[t].T)*alpha[t+1]
    expect += alpha[t].T*f[t]

# e1 = (Lu+Fu.T a)k
# e2 = (Lu+Fu.T a)Kf + (Lx+Fx a)f
    
assert( np.isclose( expect,kkt.expectedImprovement()[0]) )


### vx = qx - qxu k = qx - K qu = qx - qxu quu^-1 qu
###    = lx + Fx.T vx0 - K lu - K Fu.T vx0 = lx - K lu + (Fx-FuK.T).T vx0
###    = lx - K lu + (Fx-FuK).T (vx' - Vxx f')
###    = alpha'  +  CVxx f'
### alpha = lx - Klu + C alpha'
### Sq alpha' = vx' + d'
### alpha = lx - Klu + C alpha' = lx - Klu + C vx' + C d' = lx - Klu + C (vx'-Vxx f') + CVxxf'+ C d'
###       = vx + C(d'+Vxxf')
### => d = C(d'+Vxxf')   with C = (Fx-FuK)

d = [np.nan]*T + [ np.zeros([NX,1]) ]
for t in range(T-1,-1,-1):
    d[t] = (Fx[t]-Fu[t]*K[t]).T * (d[t+1]+Vxx[t+1]*f[t+1])
    assert( np.allclose( Vx[t]+d[t],alpha[t] ) )
    # Qu = Lu + Fu.T (Vx'-Vxx f') = Lu + Fu.T alpha' - Fu.T d' - Fu.T Vxx f'
    assert( np.allclose(Lu[t] + Fu[t].T*alpha[t+1],
                        Qu[t] + Fu[t].T*d[t+1] + Fu[t].T*Vxx[t+1]*f[t+1]))
    

### d = CVxxf' + Cd' = CVxxf' + CCVxxf'' + CCd'' = ... = CVxxf' + CCVxxf'' + ... + C^(T-t)Vxx f[T]
### cost = sum_t qu[t].T*k[t] + (Fu[t].T*d[t+1]+Fu[t].T*Vxx[t+1]*f[t+1]).T*k[t]
###               + vx[t].T*f[t] + d[t].T*f[t]
###       = sum_t  qu[t]*k[t] + vx[t]*f[t]
###                + d'.T Fuk + f'.T Vxx Fuk + d.T f
### d'.T Fuk + f'.T Vxx Fuk + d.T f = d'.T Fuk + f'.T Vxx Fuk + (d'+Vxxf').T C.T f
###    = d'.T Fuk + f'.T Vxx Fuk + f.T C d' + f.T C Vxx f'
expect0 = Vx[-1].T*f[-1]
expect0 = alpha[T].T*f[-1]
dexp = 0
for t in range(T):
    # expect0 += (Qu[t]+Fu[t].T*d[t+1]+Fu[t].T*Vxx[t+1]*f[t+1]).T*k[t] \
    #             + (Vx[t]+d[t]).T*f[t]
    expect0 += Qu[t].T*k[t] + Vx[t].T*f[t]
    #expect0 += (Lu[t] + Fu[t].T*alpha[t+1]).T*k[t] + alpha[t].T*f[t]
    dexp += d[t+1].T*Fu[t]*k[t] + f[t+1].T*Vxx[t+1]*Fu[t]*k[t] + d[t].T*f[t]
    
assert(np.isclose(expect,expect0+dexp))

k = [-ki for ki in k]
f = [-fi for fi in f]
K = [-Ki for Ki in K]
Vx = [ v+V*ft for v,V,ft in zip(Vx,Vxx,f) ]

d1,d2=[-d for d in kkt.expectedImprovement()]
dg = Vx[-1].T*f[-1]
dq =-f[-1].T*Vxx[-1]*f[-1]
dv = f[-1].T*Vxx[-1]*x[-1]
for t in range(T):
    dg += Vx[t].T*f[t]+Qu[t].T*k[t]
    dq += k[t].T*Quu[t]*k[t]-f[t].T*Vxx[t]*f[t]
    dv += f[t].T*Vxx[t]*x[t]

assert(np.isclose(d1,dg-dv))    
assert(np.isclose(d2,dq+2*dv))    

d1b = Vx[0].T*f[0] + Qu[0].T*k[0] + Vx[1].T*f[1] - f[0].T*Vxx[0]*x[0] - f[1].T*Vxx[1]*x[1]
d2b = -f[0].T*Vxx[0]*f[0] -f[1].T*Vxx[1]*f[1] + k[0].T*Quu[0]*k[0] \
      + 2*(f[0].T*Vxx[0]*x[0] + f[1].T*Vxx[1]*x[1])
