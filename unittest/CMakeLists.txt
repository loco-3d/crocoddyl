if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()

if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  file(GLOB_RECURSE ${PROJECT_NAME}_FACTORY_TEST
       ${CMAKE_CURRENT_SOURCE_DIR}/factory/*.cpp)

  if(UNIX)
    add_library(${PROJECT_NAME}_unittest SHARED ${${PROJECT_NAME}_FACTORY_TEST})
    set_target_properties(${PROJECT_NAME}_unittest PROPERTIES LINKER_LANGUAGE
                                                              CXX)
    target_link_libraries(${PROJECT_NAME}_unittest ${PROJECT_NAME}
                          example-robot-data::example-robot-data)

    if(OPENMP_FOUND)
      target_link_libraries(${PROJECT_NAME}_unittest ${OpenMP_CXX_LIBRARIES})
    endif()
  endif()

  set(${PROJECT_NAME}_CPP_TESTS
      test_states
      test_actuation
      test_activations
      test_guidance
      test_tasks
      test_residuals
      test_costs
      test_costs_noFF
      test_costs_collision
      test_cost_sum
      test_constraints
      test_constraint_manager
      test_implicit_constraints
      test_multiple_implicit_constraints
      test_kinematic_loop
      test_generic_contact
      test_multibody_actuation
      test_multibody_inertial_params
      test_parameter_energy_residuals
      test_numdiff
      test_implicit_constraint_residuals
      test_controls
      test_dynamics
      test_multibody_dynamics
      test_impulse_dynamics
      test_integrator_time
      test_integrator_timeopt_params
      test_action_integration
      test_params
      test_params_data
      test_parameter_manager
      test_observer_base
      test_integrated_observer_base
      test_observer_euler
      test_observer_rk
      test_observer_discretized
      test_lqr
      test_contact_costs
      test_impulse_costs
      test_contact_constraints
      test_impulse_constraints
      test_actions
      test_diff_actions
      test_problem
      test_problem_abstract
      test_cop_support
      test_friction_cone
      test_wrench_cone
      test_boxqp
      test_solvers)

  if(BUILD_WITH_CODEGEN_SUPPORT)
    set(${PROJECT_NAME}_CODEGEN_CPP_TESTS)
    # GCC 14+ can fail while compiling test_codegen due to a CppADCodeGen
    # thread_local guard redefinition in cppad/cg/cg.hpp.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                                VERSION_GREATER_EQUAL 14)
      message(
        WARNING
          "Skipping unittest target test_codegen with GCC ${CMAKE_CXX_COMPILER_VERSION} "
          "because of a known CppADCodeGen TLS guard redefinition issue in cppad/cg/cg.hpp."
      )
    else()
      list(APPEND ${PROJECT_NAME}_CODEGEN_CPP_TESTS test_codegen
           test_codegen_float32)
    endif()
    list(APPEND ${PROJECT_NAME}_CPP_TESTS ${${PROJECT_NAME}_CODEGEN_CPP_TESTS})
  endif()

  include_directories(.)

  macro(ADD_TEST_CFLAGS target flag)
    set_property(
      TARGET ${target}
      APPEND_STRING
      PROPERTY COMPILE_FLAGS " ${flag}")
  endmacro(ADD_TEST_CFLAGS)

  foreach(NAME ${${PROJECT_NAME}_CPP_TESTS})
    set(UNITTEST_NAME ${NAME})

    add_unit_test(${UNITTEST_NAME} ${NAME}.cpp)
    set_tests_properties(${UNITTEST_NAME} PROPERTIES TIMEOUT 7200)

    set_target_properties(${UNITTEST_NAME} PROPERTIES LINKER_LANGUAGE CXX)

    add_test_cflags(${UNITTEST_NAME} "-DBOOST_TEST_DYN_LINK")

    set(MODULE_NAME "${NAME}Test")
    string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})
    add_test_cflags(${UNITTEST_NAME} "-DTEST_MODULE_NAME=${MODULE_NAME}")
    add_test_cflags(${UNITTEST_NAME} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

    target_link_libraries(${UNITTEST_NAME} ${PROJECT_NAME}
                          ${PROJECT_NAME}_unittest)
    target_link_libraries(${UNITTEST_NAME}
                          example-robot-data::example-robot-data)

    add_test_cflags(
      ${UNITTEST_NAME}
      '-DCROCODDYL_SOURCE_DIR=\\\"${${PROJECT_NAME}_SOURCE_DIR}\\\"')
  endforeach(NAME ${${PROJECT_NAME}_CPP_TESTS})

  if(TARGET test_actions)
    target_compile_definitions(test_actions PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_residuals)
    target_compile_definitions(test_residuals PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_cost_sum)
    target_compile_definitions(test_cost_sum PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_constraint_manager)
    target_compile_definitions(test_constraint_manager
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_params_data)
    target_compile_definitions(test_params_data PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_parameter_manager)
    target_compile_definitions(test_parameter_manager
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_integrator_timeopt_params)
    target_compile_definitions(test_integrator_timeopt_params
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_action_integration)
    target_compile_definitions(test_action_integration
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_lqr)
    target_compile_definitions(test_lqr PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_numdiff)
    target_compile_definitions(test_numdiff PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_observer_euler)
    target_compile_definitions(test_observer_euler
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_observer_rk)
    target_compile_definitions(test_observer_rk PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()
  if(TARGET test_observer_discretized)
    target_compile_definitions(test_observer_discretized
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_dynamics)
    target_compile_definitions(test_dynamics PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_multibody_dynamics)
    target_compile_definitions(test_multibody_dynamics
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_impulse_dynamics)
    target_compile_definitions(test_impulse_dynamics
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_problem_abstract)
    target_compile_definitions(test_problem_abstract
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_multiple_implicit_constraints)
    target_compile_definitions(test_multiple_implicit_constraints
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_generic_contact)
    target_compile_definitions(test_generic_contact
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_multibody_actuation)
    target_compile_definitions(test_multibody_actuation
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_multibody_inertial_params)
    target_compile_definitions(test_multibody_inertial_params
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_parameter_energy_residuals)
    target_compile_definitions(test_parameter_energy_residuals
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(TARGET test_kinematic_loop)
    target_compile_definitions(test_kinematic_loop
                               PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  endif()

  if(BUILD_WITH_CODEGEN_SUPPORT)
    foreach(TEST_NAME ${${PROJECT_NAME}_CODEGEN_CPP_TESTS})
      target_link_libraries(${TEST_NAME} ${CMAKE_DL_LIBS} ${cppad_LIBRARY})
    endforeach()
  endif()

  if(BUILD_WITH_CODEGEN_SUPPORT
     AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
     AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14)
    # Safeguard in case test_codegen is created by helper macros: keep the
    # target available, but do not build it by default.
    if(TARGET test_codegen)
      set_target_properties(
        test_codegen PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD
                                                      TRUE)
    endif()
    if(TEST test_codegen)
      set_tests_properties(test_codegen PROPERTIES DISABLED TRUE)
    endif()
  endif()
endif()
