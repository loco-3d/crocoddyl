'''
This class creates an object logger with a callback operator logger(solver)
that the solver can call at every iteration end to store some data and display the robot motion
in gepetto-gui.
In the solver, set up the logger with solver.callback = SolverLogger(), and add the robot-wrapper
object in argument if you want to use the display functionalities.
'''

import copy
def displayTrajectory(robot,xs,timeline,rate=-1):
    '''
    Display a robot trajectory xs using Gepetto-viewer gui.
    '''
    if not hasattr(robot,'viewer'): robot.initDisplay(loadModel=True)
    import numpy as np
    a2m = lambda a: np.matrix(a).T
    import time
    S = 1 if rate<=0 else max(len(xs)/rate,1)
    for i,x in enumerate(xs):
        dt = timeline[i]
        if not i % S:
            robot.display(a2m(x[:robot.nq]))
        time.sleep(dt)

class SolverLogger:
    def __init__(self):
        self.steps = []
        self.iters = []
        self.costs = []
        self.control_regs = []
        self.state_regs = []
        self.th_stops = []
        self.gm_stops = []
        self.xs = []
        self.us = []
    def __call__(self,solver):
        self.xs = copy.copy(solver.xs)
        self.us = copy.copy(solver.us)
        self.steps.append( solver.stepLength )
        self.iters.append( solver.iter )
        self.costs.append(copy.copy([ d.cost for d in solver.datas() ]))
        self.control_regs.append( solver.u_reg )
        self.state_regs.append( solver.x_reg )
        self.th_stops.append(solver.stop)
        self.gm_stops.append(solver.gamma)

class SolverPrinter:
    def __init__(self,level=0):
        self.level = level
    def __call__(self,solver):
        if solver.iter % 10 == 0:
            if self.level == 0:
                print "iter \t cost \t      theta \t    gamma \t  muV \t      muLM \t alpha"
            elif self.level == 1:
                print "iter \t cost \t      theta \t    gamma \t  muV \t      muLM \t alpha \t   dV-exp \t  dV"
        if self.level == 0:
            print "%4i  %0.5e  %0.5e  %0.5e  %10.5e  %0.5e  %0.4f" % \
                (solver.iter, sum(copy.copy([ d.cost for d in solver.datas() ])),
                solver.stop, solver.gamma,
                solver.x_reg, solver.u_reg,
                solver.stepLength)
        elif self.level == 1:
            print "%4i  %0.5e  %0.5e  %0.5e  %10.5e  %0.5e  %0.4f  %0.5e  %0.5e" % \
                (solver.iter, sum(copy.copy([ d.cost for d in solver.datas() ])),
                solver.stop, solver.gamma,
                solver.x_reg, solver.u_reg,
                solver.stepLength, solver.dV_exp, solver.dV)

class SolverDisplay:
    def __init__(self,robotwrapper,rate=-1):
        self.robotwrapper = robotwrapper
        self.rate = rate
    def __call__(self,solver):
        timeline = copy.copy([ m.timeStep for m in solver.models() ])
        displayTrajectory(self.robotwrapper,solver.xs,timeline,self.rate)