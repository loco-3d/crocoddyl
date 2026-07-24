set(${PROJECT_NAME}_PYTHON_BINDINGS_TESTS
    states
    actions
    shooting
    problem_abstract
    solvers
    costs
    cost_sum
    constraint_manager
    dynamics
    multibody_dynamics
    params
    params_data
    parameter_manager
    time_parameterization
    action_integration
    lqr
    implicit_constraints
    multibody_actuation
    inertial_parameterization
    squashing
    copy)

if(BUILD_WITH_CODEGEN_SUPPORT)
  list(APPEND ${PROJECT_NAME}_PYTHON_BINDINGS_TESTS codegen_float32)
endif()

foreach(TEST ${${PROJECT_NAME}_PYTHON_BINDINGS_TESTS})
  python_build(. "test_${TEST}.py")
  add_python_unit_test("test_pybinds_${TEST}"
                       "unittest/bindings/test_${TEST}.py" "bindings/python")
  add_custom_target(
    "test_pybinds_${TEST}"
    ${CMAKE_COMMAND} -E env
    PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
    ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_${TEST}.py")
endforeach(TEST ${${PROJECT_NAME}_PYTHON_BINDINGS_TESTS})
