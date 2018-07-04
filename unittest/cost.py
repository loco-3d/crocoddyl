import unittest
import numpy as np
from models.simple_cost import GoalQuadraticCost, GoalResidualQuadraticCost, StateRunningQuadraticCost


class QuadraticCostTest(unittest.TestCase):
  def setUp(self):
    # Defined the state and control dimensions
    self.n = 4
    self.m = 2

    # Create random state and control values
    self.x = np.random.rand(self.n, 1)
    self.u = np.random.rand(self.m, 1)

    # Create desired state
    x_des = np.matrix(np.zeros((self.n, 1)))

    # Create random state and control weights
    self.q = np.random.rand(self.n)
    self.r = np.random.rand(self.m)

    # Create the different cost function and them data
    self.t_cost = GoalQuadraticCost(x_des)
    self.t_data = self.t_cost.createData(self.n)
    self.tr_cost = GoalResidualQuadraticCost(x_des)
    self.tr_data = self.tr_cost.createData(self.n)
    self.r_cost = StateRunningQuadraticCost(x_des)
    self.r_data = self.r_cost.createData(self.n, self.m)

    # Set the state or control weights
    self.t_cost.setWeights(self.q)
    self.tr_cost.setWeights(self.q)
    self.r_cost.setWeights(self.q, self.r)

  def test_terminal_quadratic_cost_terms(self):
    # Get the quadratic cost terms
    pointer = id(self.t_data)
    l = self.t_cost.l(self.t_data, self.x)
    lx = self.t_cost.lx(self.t_data, self.x)
    lxx = self.t_cost.lxx(self.t_data, self.x)

    # Compute here the quadratic cost terms for comparison
    x = self.t_cost.xr(self.t_data, self.x)
    Q = np.diag(self.q.reshape(-1))
    m_l = 0.5 * x.T * Q * x
    m_lx = Q * x
    m_lxx = Q

    # Check the terminal term values
    self.assertEqual(np.asscalar(l[0]), m_l, "Wrong cost value")
    self.assertEqual(lx.all(), m_lx.all(), "Wrong Jacobian value")
    self.assertEqual(lxx.all(), m_lxx.all(), "Wrong Hessian value")

    # Check the correct data values
    self.assertEqual(np.asscalar(self.t_data.l[0]), m_l, "Wrong data collection")
    self.assertEqual(self.t_data.lx.all(), m_lx.all(), "Wrong data collection")
    self.assertEqual(self.t_data.lxx.all(), m_lxx.all(), "Wrong data collection")

    # Check the address of the cost data
    self.assertEqual(pointer, id(self.t_data), "It has been changed the data address")

  def test_terminal_residual_quadratic_cost_terms(self):
    # Get the quadratic cost terms
    pointer = id(self.tr_data)
    l = self.tr_cost.l(self.tr_data, self.x)
    lx = self.tr_cost.lx(self.tr_data, self.x)
    lxx = self.tr_cost.lxx(self.tr_data, self.x)

    # Compute here the quadratic cost terms for comparison
    r = self.tr_cost.r(self.tr_data, self.x)
    rx = self.tr_cost.rx(self.tr_data, self.x)
    Q = np.diag(self.q.reshape(-1))
    m_l = np.asscalar(0.5 * r.T * Q * r)
    m_lx = rx.T * Q * r
    m_lxx = rx.T * Q * rx

    # Check the terminal term values
    self.assertEqual(np.asscalar(l[0]), m_l, "Wrong cost value")
    self.assertEqual(lx.all(), m_lx.all(), "Wrong Jacobian value")
    self.assertEqual(lxx.all(), m_lxx.all(), "Wrong Hessian value")

    # Check the correct data values
    self.assertEqual(np.asscalar(self.tr_data.l[0]), m_l, "Wrong data collection")
    self.assertEqual(self.tr_data.lx.all(), m_lx.all(), "Wrong data collection")
    self.assertEqual(self.tr_data.lxx.all(), m_lxx.all(), "Wrong data collection")

    # Check the address of the cost data
    self.assertEqual(pointer, id(self.tr_data), "It has been changed the data address")

  def test_running_quadratic_cost(self):
    # Since running cost depends only of the state, this is equivalent to termical cost
    # for the cost value and state-related derivatives; the control-related derivatives
    # are zero
    pointer = id(self.r_data)
    l = self.r_cost.l(self.r_data, self.x, self.u)
    lx = self.r_cost.lx(self.r_data, self.x, self.u)
    lu = self.r_cost.lu(self.r_data, self.x, self.u)
    lxx = self.r_cost.lxx(self.r_data, self.x, self.u)
    luu = self.r_cost.luu(self.r_data, self.x, self.u)
    lux = self.r_cost.lux(self.r_data, self.x, self.u)
    m_l = self.t_cost.l(self.t_data, self.x)
    m_lx = self.t_cost.lx(self.t_data, self.x)
    m_lu = np.zeros((self.m, 1))
    m_lxx = self.t_cost.lxx(self.t_data, self.x)
    m_luu = np.zeros((self.m, self.m))
    m_lux = np.matrix(np.zeros((self.m, self.n)))

    # Check the running term values
    self.assertEqual(np.asscalar(l[0]), m_l, "Wrong cost value")
    self.assertEqual(lx.all(), m_lx.all(), "Wrong state Jacobian value")
    self.assertEqual(lu.all(), m_lu.all(), "Wrong control Jacobian value")
    self.assertEqual(lxx.all(), m_lxx.all(), "Wrong state Hessian value")
    self.assertEqual(luu.all(), m_luu.all(), "Wrong control Hessian value")
    self.assertEqual(lux.all(), m_lux.all(), "Wrong state/control cost derivatives value")

    # Check the correct data values
    self.assertEqual(np.asscalar(self.r_data.l[0]), m_l, "Wrong data collection")
    self.assertEqual(self.r_data.lx.all(), m_lx.all(), "Wrong data collection")
    self.assertEqual(self.r_data.lu.all(), m_lu.all(), "Wrong data collection")
    self.assertEqual(self.r_data.lxx.all(), m_lxx.all(), "Wrong data collection")
    self.assertEqual(self.r_data.luu.all(), m_luu.all(), "Wrong data collection")
    self.assertEqual(self.r_data.lux.all(), m_lux.all(), "Wrong data collection")

    # Check the address of the cost data
    self.assertEqual(pointer, id(self.r_data), "It has been changed the data address")


if __name__ == '__main__':
  unittest.main()
