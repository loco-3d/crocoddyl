SET(${PROJECT_NAME}_PYTHON_BINDINGS_TESTS
  states
  actuations
  actions
  shooting
  solvers
  costs
  contacts
  impulses
  squashing
  copy
  )

FOREACH(TEST ${${PROJECT_NAME}_PYTHON_BINDINGS_TESTS})
  PYTHON_BUILD(. "test_${TEST}.py")
  ADD_PYTHON_UNIT_TEST("test_pybinds_${TEST}" "unittest/bindings/test_${TEST}.py" bindings/python)
  ADD_CUSTOM_TARGET("test_pybinds_${TEST}"
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
    ${PYTHON_EXECUTABLE} -c "import test_${TEST}")
ENDFOREACH(TEST ${${PROJECT_NAME}_PYTHON_BINDINGS_TESTS})
