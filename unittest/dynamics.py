import unittest
import numpy as np
import cddp


class NumDiffSpringMass(cddp.SpringMass):
  def updateLinearAppr(self, dynamicsData, x, u):
      cddp.DynamicsModel.updateLinearAppr(self, dynamicsData, x, u)

class NumDiffForwardDynamics(cddp.ForwardDynamics):
  def updateLinearAppr(self, dynamicsData, x, u):
      cddp.DynamicsModel.updateLinearAppr(self, dynamicsData, x, u)


class SpringMassDerivativesTest(unittest.TestCase):
  def setUp(self):
    # Creating the dynamic model of the system and its integrator
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    dynamics = cddp.SpringMass(integrator, discretizer)
    numdiff_dynamics = NumDiffSpringMass(integrator, discretizer)

    # Creating the data
    self.data = dynamics.createData(0., 1e-3)
    self.numdiff_data = numdiff_dynamics.createData(0., 1e-3)

    # Creating the random state and control
    x = np.random.rand(dynamics.nxImpl(), 1)
    u = np.random.rand(dynamics.nu(), 1)

    # Computing the linear approximation by analytical derivatives
    dynamics.updateLinearAppr(self.data, x, u)

    # Computing the linear approximation by NumDiff
    numdiff_dynamics.updateLinearAppr(self.numdiff_data, x, u)

  def test_numerical_linear_approximation(self):
    # Checking da/dq
    self.assertTrue(np.allclose(self.data.aq, self.numdiff_data.aq), \
      "The analytical configuration derivative is wrong.")

    # Checking da/dv
    self.assertTrue(np.allclose(self.data.av, self.numdiff_data.av), \
      "The analytical velocity derivative is wrong.")

    # Checking da/dv
    self.assertTrue(np.allclose(self.data.au, self.numdiff_data.au), \
      "The analytical control derivative is wrong.")


class ForwardDynamicsDerivativesTest(unittest.TestCase):
  def setUp(self):
    # Getting the robot model from the URDF file. Note that we use the URDF file
    # installed by binary (through sudo-apt install robotpkg-talos-data)
    import pinocchio as se3
    path = '/opt/openrobots/share/talos_data/'
    urdf = path + 'robots/talos_left_arm.urdf'
    robot = se3.robot_wrapper.RobotWrapper(urdf, path)

    # Create the dynamics and its integrator and discretizer
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    dynamics = cddp.ForwardDynamics(integrator, discretizer, robot.model)
    numdiff_dynamics = NumDiffForwardDynamics(integrator, discretizer, robot.model)

    # Creating the data
    self.data = dynamics.createData(0., 1e-3)
    self.numdiff_data = numdiff_dynamics.createData(0., 1e-3)

    # Creating the random state and control
    x = np.random.rand(dynamics.nxImpl(), 1)
    u = np.random.rand(dynamics.nu(), 1)

    # Computing the linear approximation by analytical derivatives
    dynamics.updateDynamics(self.data, x, u)
    dynamics.updateLinearAppr(self.data, x, u)

    # Computing the linear approximation by NumDiff
    numdiff_dynamics.updateLinearAppr(self.numdiff_data, x, u)

  def test_numerical_linear_approximation(self):
    # Checking da/dq
    self.assertTrue(
      np.allclose(self.data.aq, self.numdiff_data.aq, rtol=1e-3, atol=1e-5), \
      "The analytical configuration derivative is wrong.")

    # Checking da/dv
    self.assertTrue(
      np.allclose(self.data.av, self.numdiff_data.av, rtol=1e-3, atol=1e-5), \
      "The analytical velocity derivative is wrong.")

    # Checking da/dv
    self.assertTrue(
      np.allclose(self.data.au, self.numdiff_data.au, rtol=1e-3, atol=1e-5), \
      "The analytical control derivative is wrong.")

if __name__ == '__main__':
  unittest.main()
