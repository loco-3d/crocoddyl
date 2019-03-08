SET(${PROJECT_NAME}_PYTHON_TESTS
  robots
  state
  activation
  costs
  dse3
  actuation
  contacts
  actions
  dynamic_derivatives
  impact
  rk4
  solvers
  ddp_contact
  armature
  quadruped
  )

FOREACH(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
  ADD_PYTHON_UNIT_TEST("py-${TEST}" "unittest/test_${TEST}.py")
ENDFOREACH(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
