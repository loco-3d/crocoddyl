SET(${PROJECT_NAME}_PYTHON_BINDINGS_TESTS
  states
  actuations
  actions
  shooting
  solvers
  costs
  contacts
  impulses
  force_cost
  friction_cone_cost
  momemtum_cost
  impulse_com_cost
  )

FOREACH(TEST ${${PROJECT_NAME}_PYTHON_BINDINGS_TESTS})
  ADD_PYTHON_UNIT_TEST("pybinds-${TEST}" "unittest/bindings/test_${TEST}.py" "bindings/python")
ENDFOREACH(TEST ${${PROJECT_NAME}_PYTHON_BINDINGS_TESTS})
