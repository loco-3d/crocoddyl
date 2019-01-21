import pinocchio
import conf_talos_1step as conf
from time import sleep

robot = conf.robot
rmodel = robot.model
rdata = robot.data

#---------------Display Initial Trajectory--------------
if conf.DISPLAY:
  robot.initDisplay(loadModel=True)
  for x in conf.X_init:  
    robot.display(x[:robot.nq])
    sleep(0.005)
#--------------------------------------------------------


