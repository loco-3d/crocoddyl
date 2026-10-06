set(${PROJECT_NAME}_CONTROL_EXAMPLES_PYTHON
    aerial_manipulator_fwddyn
    aerial_manipulator_invdyn
    b1_leg_6d_fwddyn
    b1_leg_6d_invdyn
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
    marine_manipulator_fwddyn
    marine_manipulator_invdyn
    quadrotor_fwddyn
    quadrotor_invdyn
    quadrotor_ubound
    quadruped_gaits_fwddyn
    quadruped_gaits_invdyn
    quadruped_pose_fwddyn
    quadruped_pose_invdyn
    quadruped_walk_ubound)

set(${PROJECT_NAME}_ESTIMATION_EXAMPLES_PYTHON
    b1_leg_6d_inertialparam_fwddyn
    b1_leg_6d_inertialparam_invdyn
    biped_gaits_friction_inertialparam_fwddyn
    biped_gaits_friction_inertialparam_invdyn
    double_pendulum_inertialparam_fwddyn
    double_pendulum_inertialparam_invdyn
    manipulator_inertialparam_fwddyn
    manipulator_inertialparam_invdyn
    quadrotor_inertialparam_fwddyn
    quadrotor_inertialparam_invdyn
    quadruped_gaits_friction_inertialparam_fwddyn
    quadruped_gaits_friction_inertialparam_invdyn)
if(BUILD_WITH_CODEGEN_SUPPORT)
  list(APPEND ${PROJECT_NAME}_ESTIMATION_EXAMPLES_PYTHON
       manipulator_inertialparam_fwddyn_codegen)
endif()

set(${PROJECT_NAME}_TIMEOPT_EXAMPLES_PYTHON
    double_pendulum_timeopt_fwddyn
    double_pendulum_timeopt_invdyn
    manipulator_timeopt_fwddyn
    manipulator_timeopt_invdyn
    quadrotor_timeopt_fwddyn
    quadrotor_timeopt_invdyn
    quadruped_gait_timeopt_fwddyn
    quadruped_gait_timeopt_invdyn
    quadruped_pose_timeopt_fwddyn
    quadruped_pose_timeopt_invdyn)

install(FILES __init__.py
        DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples)
install(FILES estimation/utils.py
        DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples/estimation)

add_custom_target(run_examples)

function(add_python_example_group EXAMPLE_GROUP)
  string(TOUPPER ${EXAMPLE_GROUP} EXAMPLE_GROUP_UPPER)
  set(EXAMPLE_LIST "${PROJECT_NAME}_${EXAMPLE_GROUP_UPPER}_EXAMPLES_PYTHON")
  set(EXAMPLE_GROUP_TARGET "run_examples.${EXAMPLE_GROUP}")

  add_custom_target(${EXAMPLE_GROUP_TARGET})
  add_dependencies(run_examples ${EXAMPLE_GROUP_TARGET})

  install(
    FILES ${EXAMPLE_GROUP}/__init__.py
    DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples/${EXAMPLE_GROUP})

  foreach(EXAMPLE IN LISTS ${EXAMPLE_LIST})
    set(EXAMPLE_TARGET "${EXAMPLE_GROUP_TARGET}.${EXAMPLE}")
    set(EXAMPLE_PATH "${EXAMPLE_GROUP}/${EXAMPLE}.py")

    install(
      FILES ${EXAMPLE_PATH}
      DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples/${EXAMPLE_GROUP})
    add_custom_target(
      ${EXAMPLE_TARGET}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
        ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_PATH}"
        \${INPUT}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_GROUP}"
      USES_TERMINAL)
    add_dependencies(${EXAMPLE_GROUP_TARGET} ${EXAMPLE_TARGET})

    # Control examples are too slow in Debug mode to be used as tests.
    if(EXAMPLE_GROUP STREQUAL "control" AND ((NOT CMAKE_BUILD_TYPE STREQUAL
                                              "Debug") OR ENABLE_COVERAGE))
      add_python_unit_test("example-python-${EXAMPLE}"
                           "examples/${EXAMPLE_PATH}" "bindings/python")
    endif()
  endforeach()
endfunction()

add_python_example_group(control)
add_python_example_group(estimation)
add_python_example_group(timeopt)

add_custom_target(
  "update_logfiles"
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
  PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
  /bin/bash
  ${CMAKE_CURRENT_SOURCE_DIR}/control/log/update_logfiles.sh)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(UNIX)
    add_test(
      "test_logfiles"
      /bin/bash
      -c
      "export PYTHONPATH='${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}'; export PYTHON_EXECUTABLE='${PYTHON_EXECUTABLE}'; export LD_LIBRARY_PATH='${PROJECT_BINARY_DIR}:${PROJECT_BINARY_DIR}/unittest:$ENV{LD_LIBRARY_PATH}'; export DYLD_LIBRARY_PATH='${PROJECT_BINARY_DIR}:${PROJECT_BINARY_DIR}/unittest:$ENV{DYLD_LIBRARY_PATH}'; source \"$0\""
      ${CMAKE_CURRENT_SOURCE_DIR}/control/log/update_logfiles.sh)
    set_tests_properties("test_logfiles" PROPERTIES TIMEOUT 7200)

    add_custom_target(
      "test_logfiles"
      ${CMAKE_COMMAND}
      -E
      env
      PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
      PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      /bin/bash
      ${CMAKE_CURRENT_SOURCE_DIR}/control/log/check_logfiles.sh)
  endif()
endif()
