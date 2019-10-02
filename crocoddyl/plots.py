import numpy as np
import matplotlib.pyplot as plt

class PlotUAM:
    def __init__(self, stateTraj, controlTraj, knots, dt, d, cf, cm):
        self.stateTraj = stateTraj
        self.controlTraj = controlTraj
        self.knots = knots
        self.dt = dt
        self.d = d
        self.cf = cf
        self.cm = cm
        self.PlotDataType = PlotDataUAM(self)

    def plotMotorForces(self):
        fig, axs = plt.subplots(2,2, figsize=(15,10))
        fig.suptitle('Motor forces')
        t = self.PlotDataType.t
        axs[0, 0].plot(t,self.PlotDataType.control[:,0])
        axs[0, 0].set_title('Motor 1')
        axs[0, 1].plot(t,self.PlotDataType.control[:,1])
        axs[0, 1].set_title('Motor 2')
        axs[1, 0].plot(t,self.PlotDataType.control[:,2])
        axs[1, 0].set_title('Motor 3')
        axs[1, 1].plot(t,self.PlotDataType.control[:,3])
        axs[1, 1].set_title('Motor 4')
        return fig,axs

    def plotFlyingPlatformActuation(self):
        fig, axs = plt.subplots(1,2, figsize=(15,10))
        fig.suptitle('Motor forces')
        t = self.PlotDataType.t
        axs[0].plot(t,self.PlotDataType.M[:,0], t,self.PlotDataType.M[:,1], t,self.PlotDataType.M[:,2])
        axs[0].set_title('Moments')
        axs[0].legend(['Mx','My','Mz'])
        axs[1].plot(t,self.PlotDataType.thrust)
        axs[1].set_title('Thrust')
        return fig,axs

    # def plotArmActuation(self):
    #     fig, axs = plt.subplot(1,1figsize=(15,10))
    #     fig.suptitle('Motor forces')
    #     t = self.PlotDataType.t
    #     axs[0].plot(t,self.PlotDataType.M[:,0], t,self.PlotDataType.M[:,1], t,self.PlotDataType.M[:,2])
    #     axs[0].set_title('Moments')
    #     axs[0].legend(['M1','M2','M3','M4','M5','M6'])
    #     return fig,axs

class PlotDataUAM():
    def __init__(self, model):
        self.t = np.arange(0, model.knots*model.dt, model.dt)
        self.t_state = np.append(self.t, self.t[-1]+model.dt)
        self.control = np.vstack(model.controlTraj)
        self.thrust = self.control[:,0]+self.control[:,1]+self.control[:,2]+self.control[:,3]
        Mx = model.d*(-self.control[:,0]+self.control[:,1]+self.control[:,2]-self.control[:,3])
        My = model.d*(-self.control[:,0]+self.control[:,1]-self.control[:,2]+self.control[:,3])
        Mz = model.cm/model.cf*(-self.control[:,0]-self.control[:,1]+self.control[:,2]+self.control[:,3])

        self.M = np.zeros([np.size(self.control,0), 3])
        self.M[:, 0] = Mx
        self.M[:, 1] = My
        self.M[:, 2] = Mz
