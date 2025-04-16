set(${PROJECT_NAME}_EXAMPLES_PYTHON
    double_pendulum_fwddyn
    double_pendulum_invdyn
    quadrotor_fwddyn
    quadrotor_invdyn
    quadrotor_ubound
    arm_manipulation_fwddyn
    arm_manipulation_invdyn
    quadrupedal_gaits_fwddyn
    quadrupedal_gaits_invdyn
    quadrupedal_walk_ubound
    bipedal_gaits_fwddyn
    bipedal_gaits_invdyn
    bipedal_walk_ubound
    humanoid_manipulation
    humanoid_manipulation_ubound
    humanoid_taichi
    boxfddp_vs_boxddp)

install(FILES __init__.py
        DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples)
foreach(EXAMPLES ${${PROJECT_NAME}_EXAMPLES_PYTHON})
  install(FILES ${EXAMPLES}.py
          DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples)
  add_custom_target(
    "example-${EXAMPLES}"
    ${CMAKE_COMMAND}
    -E
    env
    PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
    ${PYTHON_EXECUTABLE}
    "${PROJECT_SOURCE_DIR}/examples/${EXAMPLES}.py"
    \${INPUT})

  # examples are too slow in Debug mode to be used as tests
  if((NOT CMAKE_BUILD_TYPE STREQUAL "Debug") OR ENABLE_COVERAGE)
    add_python_unit_test("example-python-${EXAMPLES}" "examples/${EXAMPLES}.py"
                         "bindings/python")
  endif()
endforeach(EXAMPLES ${${PROJECT_NAME}_EXAMPLES_PYTHON})

add_custom_target(
  "update_logfiles"
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
  PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
  /bin/bash
  ${CMAKE_CURRENT_SOURCE_DIR}/log/update_logfiles.sh)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(UNIX)
    add_test(
      "test_logfiles"
      ${CMAKE_COMMAND}
      -E
      env
      PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
      PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      /bin/bash
      ${CMAKE_CURRENT_SOURCE_DIR}/log/update_logfiles.sh
      &&
      /bin/bash
      ${CMAKE_CURRENT_SOURCE_DIR}/log/check_logfiles.sh)
    set_tests_properties("test_logfiles" PROPERTIES TIMEOUT 7200)

    add_custom_target(
      "test_logfiles"
      ${CMAKE_COMMAND}
      -E
      env
      PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
      PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      /bin/bash
      ${CMAKE_CURRENT_SOURCE_DIR}/log/check_logfiles.sh)
  endif()
endif()
