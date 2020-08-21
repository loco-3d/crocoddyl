'''
These classes create an object logger with a callback operator callback(solver)
that the solver can call at every iteration end to store some data and display the robot motion
in gepetto-gui.
In the solver, set up the logger with solver.callback = [CallbackName()], and add the robot-wrapper
object in argument if you want to use the display functionalities.
'''

import copy
import time

from .diagnostic import displayTrajectory


class CallbackDDPLogger:
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

    def __call__(self, solver):
        self.xs = copy.copy(solver.xs)
        self.us = copy.copy(solver.us)
        self.steps.append(solver.stepLength)
        self.iters.append(solver.iter)
        self.costs.append(copy.copy([d.cost for d in solver.datas()]))
        self.control_regs.append(solver.u_reg)
        self.state_regs.append(solver.x_reg)
        self.th_stops.append(solver.stop)
        self.gm_stops.append(-solver.expectedImprovement()[1])


class CallbackDDPVerbose:
    def __init__(self, level=0):
        self.level = level

    def __call__(self, solver):
        if solver.iter % 10 == 0:
            if self.level == 0:
                print("iter \t cost \t      stop \t    grad \t  xreg \t      ureg \t step \t feas")
            elif self.level == 1:
                print("iter \t cost \t      stop \t    grad \t  xreg \t      ureg \t step \t feas \tdV-exp \t      dV")
        if self.level == 0:
            print("%4i  %0.5e  %0.5e  %0.5e  %10.5e  %0.5e   %0.4f     %1d" %
                  (solver.iter, sum(copy.copy([d.cost for d in solver.datas()])), solver.stop, -solver.d2,
                   solver.x_reg, solver.u_reg, solver.stepLength, solver.isFeasible))
        elif self.level == 1:
            print("%4i  %0.5e  %0.5e  %0.5e  %10.5e  %0.5e   %0.4f     %1d  %0.5e  %0.5e" %
                  (solver.iter, sum(copy.copy([d.cost for d in solver.datas()])), solver.stop, -solver.d2,
                   solver.x_reg, solver.u_reg, solver.stepLength, solver.isFeasible, solver.dV_exp, solver.dV))


class CallbackSolverDisplay:
    def __init__(self, robotwrapper, rate=-1, freq=1, cameraTF=None):
        self.robotwrapper = robotwrapper
        self.rate = rate
        self.cameraTF = cameraTF
        self.freq = freq

    def __call__(self, solver):
        if (solver.iter + 1) % self.freq:
            return
        dt = solver.models()[0].timeStep
        displayTrajectory(self.robotwrapper, solver.xs, dt, self.rate, self.cameraTF)


class CallbackSolverTimer:
    def __init__(self):
        self.timings = [time.time()]

    def __call__(self, solver):
        self.timings.append(time.time() - self.timings[-1])
