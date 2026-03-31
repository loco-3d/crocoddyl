set(${PROJECT_NAME}_EXAMPLES_PYTHON
    aerial_manipulator_fwddyn
    aerial_manipulator_invdyn
    biped_gaits_fwddyn
    biped_gaits_invdyn
    biped_pose_fwddyn
    biped_pose_invdyn
    biped_walk_ubound
    boxfddp_vs_boxddp
    double_pendulum_fwddyn
    double_pendulum_invdyn
    double_pendulum_continuous_fwddyn
    double_pendulum_continuous_invdyn
    humanoid_backflip_fwddyn
    humanoid_backflip_invdyn
    humanoid_bar_fwddyn
    humanoid_bar_invdyn
    humanoid_frontflip_fwddyn
    humanoid_frontflip_invdyn
    humanoid_handstanding_fwddyn
    humanoid_handstanding_invdyn
    humanoid_manipulation_ubound
    humanoid_manipulation
    humanoid_taichi
    humanoid_monkey_bar_fwddyn
    humanoid_monkey_bar_invdyn
    humanoid_walking_on_hands_fwddyn
    humanoid_walking_on_hands_invdyn
    lqr_fwddyn
    lqr_invdyn
    manipulator_fwddyn
    manipulator_invdyn
    quadrotor_fwddyn
    quadrotor_invdyn
    quadrotor_ubound
    quadruped_gaits_fwddyn
    quadruped_gaits_invdyn
    quadruped_pose_fwddyn
    quadruped_pose_invdyn
    quadruped_walk_ubound)

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
