from utils import raiseIfNan
import numpy as np
import scipy.linalg as scl
from itertools import izip


rev_enumerate = lambda l: izip(xrange(len(l)-1, -1, -1), reversed(l))


class SolverFDDP:
    """ Run a modified version DDP solver, that is performing a more advanced
    feasibility search.

    The solver computes an optimal trajectory and control commmands by iteratives
    running backward and forward passes. The backward-pass updates locally the
    quadratic approximation of the problem and computes descent direction,
    and the forward-pass rollouts this new policy by integrating the system dynamics
    along a tuple of optimized control commands U*.
    The solver is particularly interesting when providing an unfeasible guess (ie x is not
    the rollout of u). In that case the solver does not try to immediatly compute a
    feasible candidate, but rather maintains the gap at the shooting nodes while it is
    searching for a good optimization.
    :param shootingProblem: shooting problem (list of action models along trajectory)
    """
    def __init__(self,shootingProblem):
        self.problem = shootingProblem
        self.allocate()

        self.isFeasible = False  # Change it to true if you know that datas[t].xnext = xs[t+1]
        self.alphas = [2**(-n) for n in range(10)]
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

    def models(self):
        """ Return all action models
        """
        return self.problem.runningModels + [self.problem.terminalModel]
    def datas(self):
        """ Return the data for all action models.
        """
        return self.problem.runningDatas + [self.problem.terminalData]

    def setCandidate(self,xs=None,us=None,isFeasible=False,copy=True):
        """ Set the warm-point.

        Set the solver candidate value for the decision variables, as a 
        trajectory xs,us of T+1 and T elements.
        :params isFeasible: True for xs are obtained from integrating the us (roll-out).
        :params copy: True for making a copy of the data
        """
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
        """ Compute the tangent (LQR) model.
        """
        self.cost = self.problem.calcDiff(self.xs,self.us)
        if self.isFeasible:
            if hasattr(self,'gaps'): del self.gaps
        else:
            # Gap store the displacement to go from a feasible trajectory to the current xs.
            # gap is so that    xconti [+] gap = xcand  ...
            #              ...  gap = xcand [-] xconti = DIFF(xconti,xcand)
            self.gaps = \
            [ self.problem.runningModels[0].State.diff(self.problem.initialState,self.xs[0]) ] \
            + [ m.State.diff(d.xnext,x) \
                          for m,d,x in zip(self.problem.runningModels,
                                           self.problem.runningDatas,self.xs[1:]) ]
        return self.cost
    
    def computeDirection(self,recalc=True):
        """ Compute the descent direction dx,dx.

        :params recalc: True for recalculating the derivatives at current state and control.
        :returns the descent direction dx,du and the dual lambdas as lists of
        T+1, T and T+1 lengths. 
        """
        if recalc: self.calc()
        self.backwardPass()
        return [ np.nan ]*(self.problem.T+1), self.k, self.Vx
        
    def stoppingCriteria(self):
        """ Return a sum of positive parameters whose sum quantifies the
        algorithm termination.
        """
        return  [ sum(q**2) for q in self.Qu ]
    
    def expectedImprovement(self):
        """ Return two scalars denoting the quadratic improvement model
        (i.e. dV = f_0 - f_+ = d1*a + d2*a**2/2)
        """
        d1 = sum([  np.dot(q,k)           for q,k in zip(self.Qu,self.k) ])
        d2 = sum([ -np.dot(k,np.dot(q,k)) for q,k in zip(self.Quu,self.k) ])
        return [ d1, d2 ]

    def tryStep(self,stepLength):
        """ Rollout the system with a predefined step length.

        :param stepLength: step length
        """
        self.forwardPass(stepLength)
        return self.cost - self.cost_try

    def solve(self,maxiter=100,init_xs=None,init_us=None,isFeasible=False,regInit=None):
        """ Nonlinear solver iterating over the solveQP.

        Compute the optimal xopt,uopt trajectory as lists of T+1 and T terms.
        And a boolean describing the success.
        :param maxiter: Maximum allowed number of iterations
        :param init_xs: Initial state
        :param init_us: Initial control
        """
        self.setCandidate(init_xs,init_us,isFeasible=isFeasible,copy=True)
        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin
        self.wasFeasible = False
        for i in range(maxiter):
            #print 'i',i
            try:
                self.computeDirection()
            except ArithmeticError:
                self.increaseRegularization()
            d1,d2 = self.expectedImprovement()

            for a in self.alphas:
                #print 'a',a
                try:
                    self.dV = self.tryStep(a)
                except ArithmeticError:
                    #print 'a=%0.3f,   arithmerror .... reject' % a
                    continue
                self.dV_exp = a*(d1+.5*d2*a)
                #print 'a=%0.3f,   exp=%5.1e,   actual=%5.1e .... ' % (a,self.dV_exp,self.dV)
                #or not self.isFeasible \
                if d1<self.th_grad \
                   or self.dV > self.th_acceptStep*self.dV_exp:
                    # Accept step
                    self.wasFeasible = self.isFeasible
                    self.setCandidate(self.xs_try,self.us_try,isFeasible=(self.wasFeasible or a==1),copy=False)
                    self.cost = self.cost_try
                    break
            if a>self.th_step:
                self.decreaseRegularization()
            if a==self.alphas[-1]:
                self.increaseRegularization()
                if self.x_reg==self.regMax:
                    raise ValueError('Max regularization reached')
            self.stepLength = a; self.iter = i
            self.stop = sum(self.stoppingCriteria())
            if self.callback is not None: [c(self) for c in self.callback]

            if self.wasFeasible and self.stop<self.th_stop:
                return self.xs,self.us,True
            # if d1<self.th_grad:
            #     return self.xs,self.us,False
            
        # Warning: no convergence in max iterations
        return self.xs,self.us,False

    def increaseRegularization(self):
        self.x_reg *= self.regFactor
        if self.x_reg > self.regMax: self.x_reg = self.regMax
        self.u_reg = self.x_reg
    def decreaseRegularization(self):
        self.x_reg /= self.regFactor
        if self.x_reg < self.regMin: self.x_reg = self.regMin
        self.u_reg = self.x_reg

    
    #### DDP Specific
    def allocate(self):
        """  Allocate matrix space of Q,V and K. 
        Done at init time (redo if problem change).
        """
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
        """ Run the backward-pass of the DDP algorithm.

        The backward-pass is equivalent to a Riccati recursion. It updates the
        quadratic terms of the optimal control problem, and the gradient and
        Hessian of the value function. Additionally, it computes the new
        feedforward and feedback commands (i.e. control policy). A regularization
        scheme is used to ensure a good search direction. The norm of the gradient,
        a the directional derivatives are computed.
        """
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
                assert( np.allclose(model.State.diff(xs[t+1],data.xnext),-self.gaps[t+1]) )
                relinearization = np.dot(self.Vxx[t+1],-self.gaps[t+1])
                #relinearization = np.dot(self.Vxx[t+1],model.State.diff(xs[t+1],data.xnext))
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
                print Lb
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
            
    def forwardPass(self,stepLength,warning='ignore'):
        """ Run the forward-pass of the DDP algorithm.

        The forward-pass basically applies a new policy and then rollout the
        system. After this rollouts, it's checked if this policy provides a
        reasonable improvement. For that we use Armijo condition to evaluated the
        choosen step length.
        :param stepLenght: step length
        """
        # Argument b is introduce for debug purpose.
        # Argument warning is also introduce for debug: by default, it masks the numpy warnings
        #    that can be reactivated during debug.
        xs,us = self.xs,self.us
        xtry = [ np.nan ]*(self.problem.T+1)  # Remove self.xtry after debug
        utry = [ np.nan ]*self.problem.T      # Remove self.utry after debug
        ctry = 0
        xnext = self.problem.initialState
        for t,(m,d) in enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            if self.isFeasible or stepLength == 1:
                xtry[t] = xnext.copy()
            else:
                xtry[t] = m.State.integrate(xnext,self.gaps[t]*(1-stepLength))
            utry[t] = us[t] - self.k[t]*stepLength  \
                      - np.dot(self.K[t],m.State.diff(xs[t],xtry[t]))
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                xnext,cost = m.calc(d, xtry[t],utry[t] )
            ctry += cost
            raiseIfNan([ctry,cost],ArithmeticError('forward error'))
            raiseIfNan(xnext,ArithmeticError('forward error'))
        if self.isFeasible or stepLength == 1:
            xtry[-1] = xnext.copy()
        else:
            xtry[-1] = self.problem.terminalModel.State.integrate(xnext,self.gaps[-1]*(1-stepLength))
        with np.warnings.catch_warnings() as npwarn:
            np.warnings.simplefilter(warning)
            ctry += self.problem.terminalModel.calc(self.problem.terminalData,xtry[-1])[1]
        raiseIfNan(ctry,ArithmeticError('forward error'))
        self.xs_try = xtry ; self.us_try = utry; self.cost_try = ctry
        return xtry,utry,ctry
