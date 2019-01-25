import solver
from utils import raiseIfNan
import numpy as np
import scipy.linalg as scl
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

        self.callback = None

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

    def solve(self,maxiter=100,init_xs=None,init_us=None,isFeasible=False,regInit=None):
        '''
        Nonlinear solver iterating over the solveQP.
        Return the optimum xopt,uopt as lists of T+1 and T terms, and a boolean
        describing the success.
        '''
        self.setCandidate(init_xs,init_us,isFeasible=isFeasible,copy=True)
        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin
        self.dV_exp = 0.
        self.dV = 0.
        for i in range(maxiter):
            self.backward_status = " "
            self.forward_status = " "
            if self.callback is not None:
                import time
                start = time.time()
            try:
                self.computeDirection()
            except ArithmeticError:
                self.increaseRegularization()
            d1,d2 = self.expectedImprovement()
            self.gamma = -d2

            for a in self.alphas:
                try:
                    self.dV = self.tryStep(a)
                except ArithmeticError:
                    self.forward_status = "f"
                    continue
                self.dV_exp = a*(d1+.5*d2*a)
                if d1<self.th_grad or not self.isFeasible or self.dV > self.th_acceptStep*self.dV_exp:
                    # Accept step
                    self.setCandidate(self.xs_try,self.us_try,isFeasible=True,copy=False)
                    self.cost = self.cost_try
                    break
                self.forward_status = "s"
            if a>self.th_step:
                self.decreaseRegularization()
            if a==self.alphas[-1]:
                self.increaseRegularization()
                self.backward_status = "f"
            self.stepLength = a; self.iter = i
            self.stop = sum(self.stoppingCriteria())
            if self.callback is not None: [c(self) for c in self.callback]

            if self.stop<self.th_stop:
                if self.callback is not None:
                    end = time.time()
                    print
                    print "EXIT: Optimal Solution Found in %0.4f sec." % (end-start)
                return self.xs,self.us,True
            # if d1<self.th_grad:
            #     return self.xs,self.us,False
            
        # Warning: no convergence in max iterations
        return self.xs,self.us,False

    def increaseRegularization(self):
        self.x_reg *= self.regFactor
        if self.x_reg > self.regMax: self.x_reg = self.regMax
        self.u_reg = self.x_reg
        self.backward_status = "r"
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
        self.Vxx = [ np.zeros([m.ndx    ,m.ndx      ]) for m in self.models() ]
        self.Vx  = [ np.zeros([m.ndx])                 for m in self.models() ]

        self.Q   = [ np.zeros([m.ndx+m.nu,m.ndx+m.nu]) for m in self.problem.runningModels ]
        self.q   = [ np.zeros([m.ndx+m.nu           ]) for m in self.problem.runningModels ]
        self.Qxx = [ Q[:m.ndx,:m.ndx] for m,Q in zip(self.problem.runningModels,self.Q) ]
        self.Qxu = [ Q[:m.ndx,m.ndx:] for m,Q in zip(self.problem.runningModels,self.Q) ]
        self.Qux = [ Qxu.T            for m,Qxu in zip(self.problem.runningModels,self.Qxu) ]
        self.Quu = [ Q[m.ndx:,m.ndx:] for m,Q in zip(self.problem.runningModels,self.Q) ]
        self.Qx  = [ q[:m.ndx]        for m,q in zip(self.problem.runningModels,self.q) ]
        self.Qu  = [ q[m.ndx:]        for m,q in zip(self.problem.runningModels,self.q) ]

        self.K   = [ np.zeros([ m.nu,m.ndx ]) for m in self.problem.runningModels ]
        self.k   = [ np.zeros([ m.nu       ]) for m in self.problem.runningModels ]
        
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
                if self.Quu[t].shape[0]>0:
                    Lb = scl.cho_factor(self.Quu[t])
                    self.K[t][:,:]   = scl.cho_solve(Lb,self.Qux[t])
                    self.k[t][:]     = scl.cho_solve(Lb,self.Qu [t])
                else:
                    pass
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
