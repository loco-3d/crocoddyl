import unittest

from crocoddyl import loadHyQ, loadTalos, loadTalosArm, loadTalosLegs


class RobotTestCase(unittest.TestCase):
    ROBOT = None
    NQ = None
    NV = None

    def test_nq(self):
        model = self.ROBOT.model
        self.assertEqual(model.nq, self.NQ, "Wrong nq value.")

    def test_nv(self):
        model = self.ROBOT.model
        self.assertEqual(model.nv, self.NV, "Wrong nv value.")

    def test_q0(self):
      self.assertTrue(hasattr(self.ROBOT, "q0"), "It doesn't have q0")

class TalosArmTest(RobotTestCase):
    RobotTestCase.ROBOT = loadTalosArm()
    RobotTestCase.NQ = 7
    RobotTestCase.NV = 7

class TalosArmFloatingTest(RobotTestCase):
    RobotTestCase.ROBOT = loadTalosArm(freeFloating=True)
    RobotTestCase.NQ = 14
    RobotTestCase.NV = 13

class TalosTest(RobotTestCase):
    RobotTestCase.ROBOT = loadTalos()
    RobotTestCase.NQ = 39
    RobotTestCase.NV = 38

class TalosLegsTest(RobotTestCase):
    RobotTestCase.ROBOT = loadTalosLegs()
    RobotTestCase.NQ = 19
    RobotTestCase.NV = 18

class HyQTest(RobotTestCase):
    RobotTestCase.ROBOT = loadHyQ()
    RobotTestCase.NQ = 19
    RobotTestCase.NV = 18



if __name__ == '__main__':
    unittest.main()
