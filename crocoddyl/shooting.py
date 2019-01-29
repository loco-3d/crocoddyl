class ShootingProblem:
    def __init__(self,initialState,runningModels,terminalModel):
        """ Declare a shooting problem.

        :param initialState: initial state
        :param runningModels: running action models
        :param terminalModel: terminal action model
        """
        self.T = len(runningModels)
        self.initialState = initialState
        self.runningModels = runningModels
        self.runningDatas  = [ m.createData() for m in runningModels ]
        self.terminalModel = terminalModel
        self.terminalData  = terminalModel.createData()

    def calc(self,xs,us):
        """ Compute the cost and the next states.
        """
        return sum([ m.calc(d,x,u)[1]
                     for m,d,x,u in zip(self.runningModels,self.runningDatas,xs[:-1],us)]) \
            + self.terminalModel.calc(self.terminalData,xs[-1])[1]
    def calcDiff(self,xs,us):
        """ Compute the cost-and-dynamics functions their derivatives.

        These quantities are computed along a given pair of trajectories xs
        (states) and us (controls).
        :param xs: state trajectory
        :param us: control trajectory
        """
        assert(len(xs)==self.T+1)
        assert(len(us)==self.T)
        for m,d,x,u in zip(self.runningModels,self.runningDatas,xs[:-1],us):
            m.calcDiff(d,x,u)
        self.terminalModel.calcDiff(self.terminalData,xs[-1])
        return sum([ d.cost for d in self.runningDatas+[self.terminalData] ])

    def rollout(self,us):
        """ Integrate the dynamics given a control sequence.

        :param us: control sequence
        """
        xs = [ self.initialState ]
        for m,d,u in zip(self.runningModels,self.runningDatas,us):
            xs.append( m.calc(d,xs[-1],u)[0].copy() )
        return xs