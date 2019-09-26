SET(${PROJECT_NAME}_PYTHON_TESTS
  actions
  activation
  actuation
  armature
  boxsolvers
  contacts
  costs
  ddp_contact
  dse3
  dynamic_derivatives
  impact
  quadruped
  rk4
  robots
  solvers
  state
)
if(${MULTICONTACT_API_FOUND})
  SET(${PROJECT_NAME}_PYTHON_TESTS ${${PROJECT_NAME}_PYTHON_TESTS} locomotion)
endif(${MULTICONTACT_API_FOUND})

FOREACH(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
  ADD_PYTHON_UNIT_TEST("py-${TEST}" "unittest/python/test_${TEST}.py" ".")
ENDFOREACH(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
