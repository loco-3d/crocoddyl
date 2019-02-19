import unittest
import sys

testmodules = [
  # 'test_state',
  # 'test_costs',
  # # 'test_dynamics', TODO first create dynamics abstraction
  # 'test_actuation',
  # 'test_contacts', # TODO investigate why sometimes doesn't work
  # 'test_actions',
  # 'test_solvers',
  # 'test_activation',
  # 'test_ddp_contact',
  # 'test_dse3',
  # 'test_dynamic_derivatives',
  # # 'test_impact', # TODO investigate why doesn't work
  # 'test_robots',
  # 'test_rk4',
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
