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




class IntegratedActionModelRK4:
    def __init__(self,diffModel, timestep=1e-3):
        self.differential = diffModel
        self.State = self.differential.State
        self.nx    = self.differential.nx
        self.ndx   = self.differential.ndx
        self.nu    = self.differential.nu
        self.nq    = self.differential.nq
        self.nv    = self.differential.nv
        self.timeStep = timeStep
        self.rk4_inc =[0.5,0.5,1.]
    @property
    def ncost(self): return self.differential.ncost
    def createData(self):
      return IntegratedActionDataRK4(self)
    '''
    xn = x + dt/6(k0+2k1+2k2+k3)

    dx/dt = m(x,u) = [x[nq:], ddq(x,u)]

    data.y[0]= x
    k0 = m(y[0])
    data.y[1] = x+(dt/2)k0
    k1 = m(y[1])
    data.y[2] = x+(dt/2)k1
    k2 = m(y[2])
    data.y[3] = x+(dt)k2
    k3 = m(y[3])

    data.dx = (dt/6)(k0+2k1+2k2+k3)

    '''
    def calc(model,data,x,u=None):
      nx,ndx,nu,ncost,nq,nv,dt = model.nx,model.ndx,model.nu,model.ncost,\
                                 model.nq,model.nv,model.timeStep

      data.y[0] = x
      for i in xrange(3):
        data.acc[i], data.l[i] = model.differential.calc(data.differential[i], data.y[i], u)
        data.ki[i] = np.concatenate([data.y[i][nq:], data.acc[i]])
        data.y[i+1] = model.differential.State.integrate(x,
                                                         data.ki[i]*model.rk4_inc[i]*dt)

      data.acc[3],data.l[3] = model.differential.calc(data.differential[3], data.y[3], u)
      data.ki[3] = np.concatenate([data.y[3][nq:], data.acc[3]])
      data.dx = (data.ki[0]+ 2.*data.ki[1]+ 2.*data.ki[2]+ data.ki[3])*dt/6
      data.xnext[:] = model.differential.State.integrate(x,data.dx)

      data.cost = (data.l[0]+2*data.l[1]+2*data.l[2]+data.l[3])/6

      return data.xnext, data.cost

    '''
    xn = integrate(x , dt/6(k0+2k1+2k2+k3)) = integrate(x, dx)
    -----------------------------------------------------------------
    dxn_dx = dintegrate_left + dintegrate_right*(ddx_dx)

    ddx_dx = dt/6(dk0_dx +2*dk1_dx +2*dk2_dx + dk3_dx )

    dk0_dx = dk0_dy0 * dy0_dx = dk0_dy0
    dk1_dx = dk1_dy1 * dy1_dx
    dk2_dx = dk2_dy2 * dy2_dx
    dk3_dx = dk3_dy3 * dy3_dx

    dk1_dyi = model.differential.calcdiff(yi) for all i

    dy0_dx = Identity
    dy1_dx = d(integrate(x, (dt/2)k0))_dx = dintegrate_left + dt/2*dintegrate_right*dk0_dx
    dy2_dx = d(integrate(x, (dt/2)k1))_dx = dintegrate_left + dt/2*dintegrate_right*dk1_dx
    dy3_dx = d(integrate(x, (dt)k2))_dx = dintegrate_left + dt*dintegrate_right*dk2_dx

    -----------------------------------------------------------------
    dxn_du = dintegrate_right*(ddx_du)

    ddx_du = dt/6(dk0_du +2*dk1_du +2*dk2_du + dk3_du )
    dk0_du = np.vstack([0, da_du(y[0], u) ])

    partialdk1_du = np.vstack([0, da_du(y[1], u)])
    partialdk2_du = np.vstack([0, da_du(y[2], u)])
    partialdk3_du = np.vstack([0, da_du(y[3], u)])

    dk1_du = dk1_dy1*dy1_du+ partialdk1_du
    dk2_du = dk2_dy2*dy2_du+ partialdk2_du
    dk3_du = dk3_dy3*dy3_du+ partialdk3_du

    dy0_du = 0
    dy1_du = dintegrate_right*dt/2*dk0_du
    dy2_du = dintegrate_right*dt/2*dk1_du
    dy3_du = dintegrate_right*dt*dk2_du
    '''

    def calcDiff(model,data,x,u=None,recalc=True):
        nx,ndx,nu,ncost,nq,nv,dt = model.nx,model.ndx,model.nu,model.ncost,\
                                   model.nq,model.nv,model.timeStep
        if recalc: model.calc(data,x,u)
        for i in xrange(4):
          model.differential.calcDiff(data.differential[i],data.y[i],
                                      u,recalc=False)
          data.dki_dy[i] = np.bmat([[np.zeros([nv, nv]), np.identity(nv)],
                                    [data.differential[i].Fx]])

        data.dki_du[0] = np.vstack([np.zeros([nv, nu]),
                                    data.differential[0].Fu])

        data.Lx[:] = data.differential[0].Lx
        data.Lu[:] = data.differential[0].Lu
        
        data.dy_dx[0] = np.identity(nv*2)
        data.dy_du[0] = np.zeros((ndx,nu))
        data.dki_dx[0] = data.dki_dy[0]

        data.dli_dx[0] = data.differential[0].Lx
        data.dli_du[0] = data.differential[0].Lu

        data.ddli_ddx[0] = data.differential[0].Lxx
        data.ddli_ddu[0] = data.differential[0].Luu
        data.ddli_dxdu[0] = data.differential[0].Lxu

        for i in xrange(1,4):
          c = model.rk4_inc[i-1]*dt
          dyi_dx,dyi_ddx = model.State.Jintegrate(x,c*data.ki[i-1])

          #---------Finding the derivative wrt u--------------
          data.dy_du[i] = c*np.dot(dyi_ddx, data.dki_du[i-1])
          data.dki_du[i] = np.vstack([c*data.dki_du[i-1][nv:,:],
                                      data.differential[i].Fu+\
                                      np.dot(data.differential[i].Fx, data.dy_du[i])])

          data.dli_du[i] = data.differential[i].Lu +\
                           np.dot(data.differential[i].Lx, data.dy_du[i])

          data.ddli_ddu[i] = data.differential[i].Luu +\
                             np.dot(data.dy_du[i].T, data.differential[i].Lxu) +\
                             np.dot(data.dy_du[i].T,
                                    np.dot(data.differential[i].Lxx, data.dy_du[i]))

          #---------Finding the derivative wrt x--------------
          data.dy_dx[i] = dyi_dx+c*np.dot(dyi_ddx,data.dki_dx[i-1])
          data.dki_dx[i] = np.dot(data.dki_dy[i], data.dy_dx[i])

          data.dli_dx[i] = np.dot(data.differential[i].Lx, data.dy_dx[i])
          data.ddli_ddx[i] = np.dot(data.dy_dx[i].T, np.dot(data.differential[i].Lxx,
                                                            data.dy_dx[i]))
          data.ddli_dxdu[i] = np.dot(data.dy_dx[i].T, data.differential[i].Lxu )+\
                              np.dot(data.dy_dx[i].T, np.dot(data.differential[i].Lxx,
                                                             data.dy_du[i]))

        dxnext_dx,dxnext_ddx = model.State.Jintegrate(x,data.dx)
        ddx_dx = (data.dki_dx[0]+2.*data.dki_dx[1]+2.*data.dki_dx[2]+data.dki_dx[3])*dt/6
        data.ddx_du =\
                  (data.dki_du[0]+2.*data.dki_du[1]+2.*data.dki_du[2]+data.dki_du[3])*dt/6
        data.Fx[:] = dxnext_dx + np.dot(dxnext_ddx, ddx_dx)
        data.Fu[:] = np.dot(dxnext_ddx, data.ddx_du)

        data.Lx[:] = (data.dli_dx[0]+2.*data.dli_dx[1]+2.*data.dli_dx[2]+data.dli_dx[3])/6
        data.Lu[:] = (data.dli_du[0]+2.*data.dli_du[1]+2.*data.dli_du[2]+data.dli_du[3])/6

        data.Lxx[:] = (data.ddli_ddx[0]+2.*data.ddli_ddx[1]+\
                       2.*data.ddli_ddx[2]+data.ddli_ddx[3])/6
        data.Luu[:] = (data.ddli_ddu[0]+2.*data.ddli_ddu[1]+\
                       2.*data.ddli_ddu[2]+data.ddli_ddu[3])/6
        data.Lxu[:] = (data.ddli_dxdu[0]+2.*data.ddli_dxdu[1]+\
                       2.*data.ddli_dxdu[2]+data.ddli_dxdu[3])/6
        data.Lux = data.Lxu.T
                
class IntegratedActionDataRK4:
    def __init__(self,model):
        nx,ndx,nu,ncost = model.nx,model.ndx,model.nu,model.ncost
        self.differential = [None,]*4

        for i in xrange(4):
          self.differential[i] = model.differential.createData()
        self.l = [np.nan,]*4
        self.ki = [np.zeros([ndx]),]*4
    
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

        #Quantities for derivatives
        self.dx = [np.zeros([ndx]),]*4
        self.y = [np.zeros([nx]),]*4
        self.acc = [np.zeros([nu]),]*4

        self.dki_dy = [np.zeros([ndx,ndx]),]*4
        self.dki_dx = [np.zeros([ndx,ndx]),]*4
        self.dy_dx = [np.zeros([ndx, ndx]),]*4

        self.dki_du = [np.zeros([ndx,nu]),]*4
        self.dy_du = [np.zeros([ndx,nu]),]*4
        self.ddx_du = np.zeros([ndx,nu])
        
        self.dli_dx = [np.zeros([ ndx ]),]*4
        self.dli_du = [np.zeros([ nu ]),]*4

        self.ddli_ddx = [np.zeros([ ndx,ndx ]),]*4
        self.ddli_ddu = [np.zeros([ nu,nu ]),]*4
        self.ddli_dxdu = [np.zeros([ ndx,nu ]),]*4
