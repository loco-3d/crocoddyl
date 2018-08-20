import unittest
import numpy as np
import cddp


class SpringMassNumDiffTest(unittest.TestCase):
  def setUp(self):
    # Creating the dynamic model with analytical and numerical derivatives
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    self.analytic_dyn = cddp.SpringMass(integrator, discretizer)
    self.numdiff_dyn = cddp.NumDiffSpringMass(integrator, discretizer)

    # Creating the data for both dynamic models
    self.analytic_data = self.analytic_dyn.createData()
    self.numdiff_data = self.numdiff_dyn.createData()

    # Creating the random state and control
    self.x0 = np.random.rand(self.analytic_dyn.getConfigurationDimension(), 1)
    self.u0 = np.random.rand(self.analytic_dyn.getControlDimension(), 1)

  def test_numerical_state_derivative(self):
    fx = self.analytic_dyn.fx(self.analytic_data, self.x0, self.u0)
    fx_ndiff = self.numdiff_dyn.fx(self.numdiff_data, self.x0, self.u0)
    self.assertTrue(np.allclose(fx, fx_ndiff), \
      "The state derivative computation is wrong.")

  def test_numerical_control_derivative(self):
    fu = self.analytic_dyn.fu(self.analytic_data, self.x0, self.u0)
    fu_ndiff = self.numdiff_dyn.fu(self.numdiff_data, self.x0, self.u0)
    self.assertTrue(np.allclose(fu, fu_ndiff), \
      "The control derivative computation is wrong.")


class ArmNumDiffTest(unittest.TestCase):
  def setUp(self):
    import rospkg

    # Creating the system model
    path = rospkg.RosPack().get_path('talos_data')
    urdf = path + '/robots/talos_left_arm.urdf'
    self.system = cddp.NumDiffRobotID(urdf, path)

    # Creating the system data
    self.data = self.system.createData()

  def test_numerical_state_derivative(self):
    x = np.random.rand(self.system.getConfigurationDimension(), 1)
    u = np.zeros((self.system.getControlDimension(), 1))
    
    delta_x = 1e-4 * np.ones((self.system.getConfigurationDimension(), 1))
    fx = self.system.fx(self.data, x, u).copy()
    approx_df = fx * delta_x
    df = self.system.f(self.data, x + delta_x, u).copy() -\
         self.system.f(self.data, x, u).copy()
    self.assertTrue(np.allclose(df, approx_df, atol=1e-5), \
      "The state derivative computation is wrong.")

  def test_numerical_control_derivative(self):
    x = np.zeros((self.system.getConfigurationDimension(), 1))
    u = np.random.rand(self.system.getControlDimension(), 1)
    
    delta_u = 1e-4 * np.ones((self.system.getControlDimension(), 1))
    fu = self.system.fu(self.data, x, u).copy()
    approx_df = fu * delta_u
    df = self.system.f(self.data, x, u + delta_u).copy() -\
         self.system.f(self.data, x, u).copy()
    self.assertTrue(np.allclose(df, approx_df, atol=1e-5), \
      "The control derivative computation is wrong.")

  def test_numerical_linearized_system(self):
    x = np.random.rand(self.system.getConfigurationDimension(), 1)
    u = np.random.rand(self.system.getControlDimension(), 1)

    delta_x = 1e-4 * np.ones((self.system.getConfigurationDimension(), 1))
    delta_u = 1e-4 * np.ones((self.system.getControlDimension(), 1))
    fx = self.system.fx(self.data, x, u).copy()
    fu = self.system.fu(self.data, x, u).copy()
    approx_df = fx * delta_x + fu * delta_u
    df = self.system.f(self.data, x + delta_x, u + delta_u).copy() -\
         self.system.f(self.data, x, u).copy()
    self.assertTrue(np.allclose(df, approx_df, atol=1e-5), \
      "The system linearization around the nominal point is wrong.")

if __name__ == '__main__':
  unittest.main()
