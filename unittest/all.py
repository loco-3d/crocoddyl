import unittest
import sys

testmodules = [
  # 'test_robots',
  # 'test_state',
  # 'test_activation',
  # 'test_costs',
  # 'test_dse3',
  # 'test_actuation',
  # 'test_contacts',
  # 'test_actions',
  # 'test_dynamic_derivatives',
  # 'test_impact',
  # 'test_rk4',
  # 'test_solvers',
  # 'test_ddp_contact',
  # 'test_armature',
  # 'test_quadruped',
]

suite = unittest.TestSuite()

for t in testmodules:
  try:
    # If the module defines a suite() function, call it to get the suite.
    mod = __import__(t, globals(), locals(), ['suite'])
    suitefn = getattr(mod, 'suite')
    suite.addTest(suitefn())
  except (ImportError, AttributeError):
    # else, just load all the test cases from the module.
    suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

result = unittest.TextTestRunner().run(suite)
sys.exit(len(result.errors) + len(result.failures))
