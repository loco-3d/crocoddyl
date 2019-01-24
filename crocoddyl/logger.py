'''
This class creates an object logger with a callback operator logger(solver)
that the solver can call at every iteration end to store some data and display the robot motion
in gepetto-gui.
In the solver, set up the logger with solver.callback = SolverLogger(), and add the robot-wrapper
object in argument if you want to use the display functionalities.
'''

import copy
def disptraj(robot,xs,dt=0.1,N=-1):
    '''
    Display a robot trajectory xs using Gepetto-viewer gui.
    '''
    if not hasattr(robot,'viewer'): robot.initDisplay(loadModel=True)
    import numpy as np
    a2m = lambda a: np.matrix(a).T
    import time
    S = 1 if N<=0 else max(len(xs)/N,1)
    for i,x in enumerate(xs):
        if not i % S:
            robot.display(a2m(x[:robot.nq]))
            time.sleep(dt)

class SolverLogger:
    def __init__(self,robotwrapper=None):
        self.steps = []
        self.iters = []
        self.costs = []
        self.regularizations = []
        self.robotwrapper = robotwrapper
        self.xs = []
        self.us = []
    def __call__(self,solver):
        self.xs = copy.copy(solver.xs)
        self.us = copy.copy(solver.us)
        self.steps.append( solver.stepLength )
        self.iters.append( solver.iter )
        self.costs.append( [ d.cost for d in solver.datas() ] )
        self.regularizations.append( solver.x_reg )
        if self.robotwrapper is not None:
            disptraj(self.robotwrapper,solver.xs,dt=1e-3,N=3)
