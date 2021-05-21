IF(BUILD_PYTHON_INTERFACE)
  SET(${PROJECT_NAME}_EXAMPLES_PYTHON
    double_pendulum
    boxfddp_vs_boxddp
    quadrotor
    quadrotor_ubound
    arm_manipulation
    quadrupedal_gaits
    quadrupedal_walk_ubound
    bipedal_walk
    bipedal_walk_ubound
    humanoid_manipulation
    humanoid_manipulation_ubound
    humanoid_taichi
    )

  INSTALL(FILES __init__.py DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples)
  FOREACH(examples ${${PROJECT_NAME}_EXAMPLES_PYTHON})
    PYTHON_BUILD(. "${examples}.py")
    INSTALL(FILES ${examples}.py DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples)
    ADD_CUSTOM_TARGET("examples-${examples}"
      ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
      ${PYTHON_EXECUTABLE} -c "import ${examples}" \${INPUT})

    # examples are too slow in Debug mode to be used as tests
    IF(CMAKE_BUILD_TYPE STREQUAL "Release")
      ADD_PYTHON_UNIT_TEST("example-python-${examples}" "examples/${examples}.py" bindings/python)
    ENDIF(CMAKE_BUILD_TYPE STREQUAL "Release")
  ENDFOREACH(examples ${${PROJECT_NAME}_EXAMPLES_PYTHON})
ENDIF(BUILD_PYTHON_INTERFACE)
