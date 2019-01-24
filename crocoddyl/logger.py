import shooting


class SolverAbstract:
    def __init__(self,shootingProblem):
        self.problem = shootingProblem
    
        self.alphas = [ 10**(-n) for n in range(7) ]
        self.th_acceptStep = .1
        self.th_stop = 1e-12
        
    def models(self) : return self.problem.runningModels + [self.problem.terminalModel]
    def datas(self) : return self.problem.runningDatas + [self.problem.terminalData]

    def setCandidate(self,xs=None,us=None,isFeasible=False,copy=True):
        '''
        Set a candidate pair xs,us, and define it as feasible (i.e. obtained from
        rollout) or not.
        '''
        pass
    def calc(self):
        '''
        Compute the tangent (LQR) model.
        Returns cost.
        '''
        self.cost = problem.calcDiff(xs,us)
        return self.cost
        
    def computeDirection(self,recalc=True):
        '''
        Compute the descent direction dx,dx.
        Returns the descent direction dx,du and the dual lambdas as lists of T+1, T and T+1 lengths. 
        '''
        if recalc: self.calc()
        return # xs,us,lambdas

    def tryStep(self,stepLength):
        '''
        Make a step of length stepLength in the direction self.dxs,self.dus computed 
        (possibly partially) by computeDirection. Store the result in self.xs_try,self.us_try.
        Return the cost improvement, i.e. self.cost-self.problem.calc(self.xs_try,self.us_try).
        '''
        # self.xs_try,self.us_try = ...
        # dV = self.cost - ...
        return # dV
            
    def stoppingCriteria(self):
        '''
        Return a list of positive parameters whose sum quantifies the algorithm termination.
        '''
        return    #[ criterion1, criterion2 ... ]
    
    def expectedImprovement(self):
        '''
        Return two scalars denoting the quadratic improvement model
        (i.e. dV = f_0 - f_+ = d1*a + d2*a**2/2)
        '''
        return # [ d1, d2 ]

    def solver(self,maxiter=100,init_xs=None,init_us=None):
        '''
        Nonlinear solver iterating over the solveQP.
        Return the optimum xopt,uopt as lists of T+1 and T terms, and a boolean
        describing the success.
        '''


import copy
class SolverLogger:
    def __init__(self):
        self.steps = []
        self.iters = []
        self.costs = []
        self.regularizations = []
        self.xs = []
        self.us = []
    def __call__(self,solver):
        self.xs.append(copy.copy(solver.xs))
        self.steps.append( solver.stepLength )
        self.iters.append( solver.iter )
        self.costs.append( [ d.cost for d in solver.datas() ] )
        self.regularizations.append( solver.x_reg )
