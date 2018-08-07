import numpy as np
import pinocchio as se3
import cddp


class Arm(cddp.NumDiffDynamicalSystem):
  def __init__(self, urdf, path):
    # Getting the Pinocchio model of the robot
    self.robot = se3.robot_wrapper.RobotWrapper(urdf, path)
    self.rmodel = self.robot.model
    self.rdata = self.robot.data

    # Initializing the dynamic model with numerical differentiation
    nq = self.robot.nq + self.robot.nv
    nv = self.robot.nv + self.robot.nv
    m = self.robot.nv
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    cddp.NumDiffDynamicalSystem.__init__(self, nq, nv, m, integrator, discretizer)
  
  def f(self, data, x, u):
    q = x[:self.robot.nq]
    v = x[self.robot.nq:]
    se3.aba(self.rmodel, self.rdata, q, v, u)
    data.f[:self.robot.nq] = v
    data.f[self.robot.nq:] = self.rdata.ddq
    return data.f

  def computePerturbedConfiguration(self, x, index):
    if index < self.robot.nv:
      x_pert = x.copy()
      v_pert = np.zeros((self.robot.nv, 1))
      v_pert[index] += self.sqrt_eps

      q = x[:self.robot.nq]
      x_pert[:self.robot.nq] = se3.integrate(self.rmodel, q, v_pert)
    else:
      # A perturbation in the tangent manifold has the same effect in the
      # configuration manifold because it's a classical system.
      x_pert = x.copy()
      x_pert[index] += self.sqrt_eps
    return x_pert











# np.set_printoptions(linewidth=400, suppress=True, threshold=np.nan)
# import rospkg
# path = rospkg.RosPack().get_path('talos_data')
# urdf = path + '/robots/talos_left_arm.urdf'

# dynamics = Arm(urdf, path)
# x0 = np.zeros((dynamics.getConfigurationDimension(), 1))
# u0 = np.random.rand(dynamics.getControlDimension(), 1)
# x0[0:7] = np.array([ [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7] ])
# # x0[10] = 2.

# data = dynamics.createData()
# # print dynamics.f(data, x0, u0)
# # fx, fu = dynamics.computeDerivatives(data, x0, u0, 0.01)
# fx = dynamics.fx(data, x0, u0)
# fu = dynamics.fu(data, x0, u0)
# # print fx
# # print fu
# # print u0

# delta_x = 0.001*np.ones((dynamics.getConfigurationDimension(), 1))
# delta_u = 0.001*np.ones((dynamics.getControlDimension(), 1))
# f0 = fx * delta_x + fu * delta_u
# f1 = dynamics.f(data, x0, u0).copy()
# f2 = dynamics.f(data, x0 + delta_x, u0 + delta_u).copy()
# print f0.T 
# print (f2 - f1).T
# print (f0 - (f2 - f1)).T




# # class SE3Task


# class SE3RunningCost(cddp.RunningResidualQuadraticCost):
#   def __init__(self, robot, ee_frame, M_des):
#     self.robot = robot
#     self._frame_idx = self.robot.model.getFrameId(ee_frame)
#     self.M_des = M_des
#     cddp.RunningResidualQuadraticCost.__init__(self, 6)
  
#   def r(self, data, x, u):
#     q = x[:self.robot.nq]
#     np.copyto(data.r,
#       se3.log(self.M_des.inverse() * self.robot.framePosition(q, self._frame_idx)).vector)
#     return data.r
  
#   def rx(self, data, x, u):
#     q = x[:self.robot.nq]
#     data.rx[:, :self.robot.nq] = \
#       se3.jacobian(self.robot.model, self.robot.data, q,
#                   self.robot.model.frames[self._frame_idx].parent, False, True)
#     return data.rx

#   def ru(self, data, x, u):
#     return data.ru

# frame_name = 'gripper_left_joint'
# M_des = se3.SE3(np.eye(3), np.array([ [0.], [0.], [0.5] ]))
# se3_cost = SE3RunningCost(dynamics.robot, frame_name, M_des)
# cost_data = se3_cost.createData(data.nv, data.m)
# se3_cost.r(cost_data, 0.*x0, u0)
# print se3_cost.rx(cost_data, 0.*x0, u0)

# w_se3 = np.ones(6)
# se3_cost.setWeights(w_se3)





# from simple_cost import StateControlQuadraticRegularization
# cost_manager = cddp.CostManager()


# wx = 1e-4 * np.hstack([ np.zeros(dynamics.robot.nq), np.ones(dynamics.robot.nv) ])
# wu = 1e-4 * np.ones(dynamics.getControlDimension())
# xu_reg = StateControlQuadraticRegularization()
# xu_reg.setWeights(wx, wu)

# cost_manager.addRunning(xu_reg)
# cost_manager.addRunning(se3_cost)

# timeline = np.arange(0.0, 1., 0.01)  # np.linspace(0., 0.5, 51)
# # ddp = cddp.DDP(dynamics, cost_manager, timeline)
# # ddp.compute(0.*x0)