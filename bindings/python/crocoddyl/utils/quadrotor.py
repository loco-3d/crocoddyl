import crocoddyl
import pinocchio
import numpy as np


class ActuationModelUAM(crocoddyl.ActuationModelAbstract):
    def __init__(self, state, quadrotorType, rotorDistance, coefM, coefF, uLim, lLim):
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv - 2)
        # quadrotorType (from top view)
        # X Type -> Motor 1: Front Right, CCW
        #           Motor 2: Back Left, CCW
        #           Motor 3: Front Left, CW
        #           Motor 4: Back Right, CW
        # + Type -> Motor 1: Front, CCW
        #           Motor 2: Left, CCW
        #           Motor 3: Back, CW
        #           Motor 4: Right, CW
        self.type = quadrotorType
        self.d = rotorDistance
        self.cm = coefM
        self.cf = coefF
        self.uLim = uLim
        self.lLim = lLim

        # Jacobian of generalized torque with respect motor vertical forces
        self.S = pinocchio.utils.zero((state.nv, self.nu))
        if self.type == 'x':
            self.S[2:6, :4] = np.matrix(
                [[1, 1, 1, 1], [-self.d, self.d, self.d, -self.d], [-self.d, self.d, -self.d, self.d],
                 [-self.cm / self.cf, -self.cm / self.cf, self.cm / self.cf, self.cm / self.cf]])
        elif self.type == '+':
            self.S[2:6, :4] = np.matrix(
                [[1, 1, 1, 1], [0, self.d, 0, -self.d], [-self.d, 0, self.d, 0],
                 [-self.cm / self.cf, self.cm / self.cf, -self.cm / self.cf, self.cm / self.cf]])

        # In case it is a UAM instead of a UAV, for the arm tau = u
        np.fill_diagonal(self.S[6:, 4:], 1)

    def calc(self, data, x, u):
        data.tau = self.S * u

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        data.dtau_du = self.S


class PlotUAM:
    def __init__(self, quadrotorType, stateTraj, controlTraj, knots, dt, d, cf, cm):
        self.stateTraj = stateTraj
        self.controlTraj = controlTraj
        self.knots = knots
        self.dt = dt
        self.type = quadrotorType
        self.d = d
        self.cf = cf
        self.cm = cm
        self.PlotDataType = PlotDataUAM(self)

    def plotFlyingPlatformState(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        fig.suptitle('Motor forces')
        t = self.PlotDataType.t_state
        axs[0].plot(t, self.PlotDataType.state[:, 0], t, self.PlotDataType.state[:, 1], t,
                    self.PlotDataType.state[:, 2])
        axs[0].legend(['x', 'y', 'z'])
        axs[0].set_title('Position')
        axs[1].plot(t, self.PlotDataType.state[:, 7], t, self.PlotDataType.state[:, 8], t,
                    self.PlotDataType.state[:, 9])
        axs[1].legend(['x', 'y', 'z'])
        axs[1].set_title('Velocity')
        return fig, axs

    def plotMotorForces(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Motor forces')
        t = self.PlotDataType.t
        axs[0, 0].plot(t, self.PlotDataType.control[:, 0])
        axs[0, 0].set_title('Motor 1')
        axs[0, 1].plot(t, self.PlotDataType.control[:, 1])
        axs[0, 1].set_title('Motor 2')
        axs[1, 0].plot(t, self.PlotDataType.control[:, 2])
        axs[1, 0].set_title('Motor 3')
        axs[1, 1].plot(t, self.PlotDataType.control[:, 3])
        axs[1, 1].set_title('Motor 4')
        return fig, axs

    def plotFlyingPlatformActuation(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        fig.suptitle('Motor forces')
        t = self.PlotDataType.t
        axs[0].plot(t, self.PlotDataType.M[:, 0], t, self.PlotDataType.M[:, 1], t, self.PlotDataType.M[:, 2])
        axs[0].set_title('Moments')
        axs[0].legend(['Mx', 'My', 'Mz'])
        axs[1].plot(t, self.PlotDataType.thrust)
        axs[1].set_title('Thrust')
        return fig, axs

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
        self.t = np.arange(0, round(model.knots * model.dt, 4), model.dt)
        self.t_state = np.append(self.t, self.t[-1] + model.dt)
        self.control = np.vstack(model.controlTraj)
        self.state = np.vstack(model.stateTraj)
        self.thrust = self.control[:, 0] + self.control[:, 1] + self.control[:, 2] + self.control[:, 3]
        if model.type == 'x':
            Mx = model.d * (-self.control[:, 0] + self.control[:, 1] + self.control[:, 2] - self.control[:, 3])
            My = model.d * (-self.control[:, 0] + self.control[:, 1] - self.control[:, 2] + self.control[:, 3])
            Mz = model.cm / model.cf * (-self.control[:, 0] - self.control[:, 1] + self.control[:, 2] +
                                        self.control[:, 3])
        elif model.type == '+':
            Mx = model.d * (self.control[:, 1] - self.control[:, 3])
            My = model.d * (-self.control[:, 0] + self.control[:, 2])
            Mz = model.cm / model.cf * (-self.control[:, 0] + self.control[:, 1] - self.control[:, 2] +
                                        self.control[:, 3])
        self.M = np.zeros([np.size(self.control, 0), 3])
        self.M[:, 0] = Mx
        self.M[:, 1] = My
        self.M[:, 2] = Mz
