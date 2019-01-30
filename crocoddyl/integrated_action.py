import numpy as np



class IntegratedActionModelEuler:
    def __init__(self,diffModel,timeStep=1e-3,withCostResiduals = True):
        self.differential = diffModel
        self.State = self.differential.State
        self.nx    = self.differential.nx
        self.ndx   = self.differential.ndx
        self.nu    = self.differential.nu
        self.nq    = self.differential.nq
        self.nv    = self.differential.nv
        self.withCostResiduals = withCostResiduals
        self.timeStep = timeStep
    @property
    def ncost(self): return self.differential.ncost
    def createData(self): return IntegratedActionDataEuler(self)
    def calc(model,data,x,u=None):
        nx,ndx,nu,ncost,nq,nv,dt = model.nx,model.ndx,model.nu,model.ncost,model.nq,model.nv,model.timeStep
        acc,cost = model.differential.calc(data.differential,x,u)
        if model.withCostResiduals:
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
