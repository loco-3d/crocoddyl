import numpy as np


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

    def computePrimalDual(self):
        self.primaldual = np.linalg.solve(self.kkt,-self.kktref)
        self.primal = self.primaldual[:self.ndx+self.nu]
        self.dual = self.primaldual[-self.ndx:]
        
    def computeDirection(self,recalc=True):
        '''
        Compute the direction of descent for the current guess xs,us. 
        if recalc is True, run self.calc() before hand. self.setCandidate 
        must have been called before.
        '''
        if recalc: self.calc()
        self.computePrimalDual()

        p_x  = self.primaldual[:self.ndx]
        p_u  = self.primaldual[self.ndx:self.ndx+self.nu]
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
            if verbose: print( 'Accept iter=%d, a=%f, cost=%.8f'
                               % (i,a,self.problem.calc(self.xs,self.us)))
                
            self.stop = sum(self.stoppingCriteria())
            if self.stop<self.th_stop:
                return self.xs,self.us,True
            # if d1<self.th_grad:
            #     return self.xs,self.us,False

        # Warning: no convergence in max iterations
        return self.xs,self.us,False
