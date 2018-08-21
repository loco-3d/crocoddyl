import unittest
import numpy as np
import cddp


class StateLogBarrierTest(unittest.TestCase):
  def setUp(self):
    # Dimension of our custom problem
    self.n = 3
    self.m = 2

  def test_state_barrier_in_origin(self):
    # Defining the state barrier with bounds equals 1 and -1
    ub = np.ones((self.n, 1))
    lb = -1. * ub
    logb = cddp.StateBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the barrier value in the origing
    x = np.zeros((self.n, 1))
    u = np.zeros((self.m, 1))
    l = logb.l(data, x, u)
    self.assertEqual(np.asscalar(l[0]), 0., "Barrier isn't 0.")

  def test_state_barrier_infeasible(self):
    # Defining the state barrier with bounds equals 1 and -1
    ub = np.ones((self.n, 1))
    lb = -1. * ub
    logb = cddp.StateBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the barrier value in the infeasible region
    x = 10. * np.ones((self.n, 1))
    u = np.zeros((self.m, 1))
    l = logb.l(data, x, u)
    self.assertEqual(np.asscalar(l[0]), np.finfo(float).max, "Barrier isn't inf.")

  def test_state_jacobian_signed_in_lower_bound(self):
    # Defining the state barrier with bounds equals 1 and -1
    ub = np.ones((self.n, 1))
    lb = -1. * ub
    logb = cddp.StateBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the state Jacobian in the lowe bound
    x = -np.ones((self.n, 1))
    u = np.zeros((self.m, 1))
    logb.l(data, x, u)
    lx = logb.lx(data, x, u)
    self.assertTrue((lx < 0).all(), "The Jacobian isn't negative define.")

  def test_state_jacobian_signed_in_upper_bound(self):
    # Defining the state barrier with bounds equals 1 and -1
    ub = np.ones((self.n, 1))
    lb = -1. * ub
    logb = cddp.StateBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the state Jacobian in the upper bound
    x = np.ones((self.n, 1))
    u = np.zeros((self.m, 1))
    logb.l(data, x, u)
    lx = logb.lx(data, x, u)
    self.assertTrue((lx > 0).all(), "The Jacobian isn't positive define.")

  def test_state_barrier_is_convex(self):
    # Defining the state barrier with bounds equals 1 and -1
    ub = np.random.rand(self.n, 1)
    lb = -1. * ub
    logb = cddp.StateBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the state Hessian of the barrier function
    x = np.random.uniform(0., 0.5, (self.n, 1))
    u = np.zeros((self.m, 1))
    
    logb.l(data, x, u)
    logb.lx(data, x, u)
    lxx = logb.lxx(data, x, u)
    self.assertTrue((lxx > 0).all(), "Barrier isn't convex.")

  def test_control_barrier_in_origin(self):
    # Defining the control barrier with bounds equals 1 and -1
    ub = np.ones((self.m, 1))
    lb = -1. * ub
    logb = cddp.ControlBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the barrier value in the origin
    x = np.zeros((self.n, 1))
    u = np.zeros((self.m, 1))
    l = logb.l(data, x, u)
    self.assertEqual(np.asscalar(l[0]), 0., "Barrier isn't 0.")

  def test_control_barrier_infeasible(self):
    # Defining the control barrier with bounds equals 1 and -1
    ub = np.ones((self.m, 1))
    lb = -1. * ub
    logb = cddp.ControlBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the barrier value in the infeasible region
    x = np.zeros((self.n, 1))
    u = 10. * np.ones((self.m, 1))
    l = logb.l(data, x, u)
    self.assertEqual(np.asscalar(l[0]), np.finfo(float).max, "Barrier isn't inf.")

  def test_control_jacobian_signed_in_lower_bound(self):
    # Defining the control barrier with bounds equals 1 and -1
    ub = np.ones((self.m, 1))
    lb = -1. * ub
    logb = cddp.ControlBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the control Jacobian in the lower bound
    x = np.zeros((self.n, 1))
    u = -np.ones((self.m, 1))
    logb.l(data, x, u)
    lu = logb.lu(data, x, u)
    self.assertTrue((lu < 0).all(), "The Jacobian isn't negative define.")

  def test_control_jacobian_signed_in_upper_bound(self):
    # Defining the control barrier with bounds equals 1 and -1
    ub = np.ones((self.m, 1))
    lb = -1. * ub
    logb = cddp.ControlBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the control Jacobian in the upper bound
    x = np.zeros((self.n, 1))
    u = np.ones((self.m, 1))
    logb.l(data, x, u)
    lu = logb.lu(data, x, u)
    self.assertTrue((lu > 0).all(), "The Jacobian isn't positive define.")

  def test_control_barrier_is_convex(self):
    # Defining the control barrier with bounds equals 1 and -1
    ub = np.random.rand(self.m, 1)
    lb = -1. * ub
    logb = cddp.ControlBarrier(ub, lb)
    data = logb.createData(self.n, self.m)

    # Computing the control Hessian of the barrier function
    x = np.zeros((self.n, 1))
    u = np.random.uniform(0., 0.5, (self.m, 1))
    
    logb.l(data, x, u)
    logb.lu(data, x, u)
    luu = logb.luu(data, x, u)
    self.assertTrue((luu > 0).all(), "Barrier isn't convex.")


if __name__ == '__main__':
  unittest.main()