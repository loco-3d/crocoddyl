class ShootingProblem:
    def __init__(self,initialState,runningModels,terminalModel):
        self.T = len(runningModels)
        self.initialState = initialState
        self.runningModels = runningModels
        self.runningDatas  = [ m.createData() for m in runningModels ]
        self.terminalModel = terminalModel
        self.terminalData  = terminalModel.createData()

    def calc(self,xs,us):
        return sum([ m.calc(d,x,u)[1]
                     for m,d,x,u in zip(self.runningModels,self.runningDatas,xs[:-1],us)]) \
            + self.terminalModel.calc(self.terminalData,xs[-1])[1]
    def calcDiff(self,xs,us):
        '''
        Compute the cost-and-dynamics functions-and-derivatives
        along a given pair of trajectories xs (states) and us (controls).
        '''
        assert(len(xs)==self.T+1)
        assert(len(us)==self.T)
        for m,d,x,u in zip(self.runningModels,self.runningDatas,xs[:-1],us):
            m.calcDiff(d,x,u)
        self.terminalModel.calcDiff(self.terminalData,xs[-1])
        return sum([ d.cost for d in self.runningDatas+[self.terminalData] ])
        
    def rollout(self,us):
        '''
        For a given control trajectory, integrate the dynamics from self.initialState.
        '''
        xs = [ self.initialState ]
        for m,d,u in zip(self.runningModels,self.runningDatas,us):
            xs.append( m.calc(d,xs[-1],u)[0].copy() )
        return xs