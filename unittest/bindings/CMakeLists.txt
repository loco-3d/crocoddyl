SET(${PROJECT_NAME}_PYTHON_BINDINGS_TESTS
  states
  actions
  shooting
  solvers
  )

FOREACH(TEST ${${PROJECT_NAME}_PYTHON_BINDINGS_TESTS})
  ADD_PYTHON_UNIT_TEST("pybinds-${TEST}" "unittest/bindings/test_${TEST}.py" "bindings/python")
ENDFOREACH(TEST ${${PROJECT_NAME}_PYTHON_BINDINGS_TESTS})
