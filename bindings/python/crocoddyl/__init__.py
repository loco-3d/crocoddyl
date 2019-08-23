from .libcrocoddyl_pywrap import *
from .libcrocoddyl_pywrap import __version__


def setGepettoViewerBackground(robot):
    if not hasattr(robot, 'viewer'):
        # Spawn robot model
        robot.initDisplay(loadModel=True)
        # Set white background and floor
        window_id = robot.viewer.gui.getWindowID('python-pinocchio')
        robot.viewer.gui.setBackgroundColor1(window_id, [1., 1., 1., 1.])
        robot.viewer.gui.setBackgroundColor2(window_id, [1., 1., 1., 1.])
        robot.viewer.gui.addFloor('hpp-gui/floor')
        robot.viewer.gui.setScale('hpp-gui/floor', [0.5, 0.5, 0.5])
        robot.viewer.gui.setColor('hpp-gui/floor', [0.7, 0.7, 0.7, 1.])
        robot.viewer.gui.setLightingMode('hpp-gui/floor', 'OFF')


def displayTrajectory(robot, xs, dt=0.1, rate=-1, cameraTF=None):
    """  Display a robot trajectory xs using Gepetto-viewer gui.

    :param robot: Robot wrapper
    :param xs: state trajectory
    :param dt: step duration
    :param rate: visualization rate
    :param cameraTF: camera transform
    """
    setGepettoViewerBackground(robot)
    if cameraTF is not None:
        robot.viewer.gui.setCameraTransform(0, cameraTF)
    import numpy as np

    import time
    S = 1 if rate <= 0 else max(len(xs) / rate, 1)
    for i, x in enumerate(xs):
        if not i % S:
            robot.display(x[:robot.nq])
            time.sleep(dt)


class CallbackSolverDisplay(libcrocoddyl_pywrap.CallbackAbstract):
    def __init__(self, robotwrapper, rate=-1, freq=1, cameraTF=None):
        libcrocoddyl_pywrap.CallbackAbstract.__init__(self)
        self.robotwrapper = robotwrapper
        self.rate = rate
        self.cameraTF = cameraTF
        self.freq = freq

    def __call__(self, solver):
        if (solver.iter + 1) % self.freq:
            return
        dt = solver.models()[0].dt
        displayTrajectory(self.robotwrapper, solver.xs, dt, self.rate, self.cameraTF)