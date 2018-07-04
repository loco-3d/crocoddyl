import unittest
import numpy as np
from models.spring_mass import SpringMass, NumDiffSpringMass



class NumDiffDynamicsTest(unittest.TestCase):
  def setUp(self):
    # Creating the dynamic model with analytical and numerical derivatives
    self.analytic_dyn = SpringMass()
    self.numdiff_dyn = NumDiffSpringMass()

    # Creating the data for both dynamic models
    self.analytic_data = self.analytic_dyn.createData()
    self.numdiff_data = self.numdiff_dyn.createData()

    # Creating the random state and control
    self.x0 = np.random.rand(self.analytic_dyn.getConfigurationDimension(), 1)
    self.u0 = np.random.rand(self.analytic_dyn.getControlDimension(), 1)

  def test_numerical_state_derivative(self):
    fx = self.analytic_dyn.fx(self.analytic_data, self.x0, self.u0)
    fx_ndiff = self.numdiff_dyn.fx(self.numdiff_data, self.x0, self.u0)
    self.assertEqual(fx.all(), fx_ndiff.all(), "The state derivative computation is wrong.")

  def test_numerical_control_derivative(self):
    fu = self.analytic_dyn.fu(self.analytic_data, self.x0, self.u0)
    fu_ndiff = self.numdiff_dyn.fu(self.numdiff_data, self.x0, self.u0)
    self.assertEqual(fu.all(), fu_ndiff.all(), "The control derivative computation is wrong.")


if __name__ == '__main__':
      unittest.main()
