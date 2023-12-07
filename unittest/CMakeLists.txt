if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()

file(GLOB_RECURSE ${PROJECT_NAME}_FACTORY_TEST
     ${CMAKE_CURRENT_SOURCE_DIR}/factory/*.cpp)

if(UNIX)
  add_library(${PROJECT_NAME}_unittest SHARED ${${PROJECT_NAME}_FACTORY_TEST})
  set_target_properties(${PROJECT_NAME}_unittest PROPERTIES LINKER_LANGUAGE CXX)
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
    test_residuals
    test_costs
    test_costs_noFF
    test_costs_collision
    test_cost_sum
    test_constraints
    test_constraint_manager
    test_contacts
    test_controls
    test_impulses
    test_multiple_contacts
    test_multiple_impulses
    test_contact_costs
    test_impulse_costs
    test_contact_constraints
    test_impulse_constraints
    test_actions
    test_diff_actions
    test_problem
    test_cop_support
    test_friction_cone
    test_wrench_cone
    test_boxqp
    test_solvers)

if(BUILD_WITH_CODEGEN_SUPPORT)
  set(${PROJECT_NAME}_CODEGEN_CPP_TESTS test_codegen)
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
  target_link_libraries(${UNITTEST_NAME} example-robot-data::example-robot-data)

  add_test_cflags(
    ${UNITTEST_NAME}
    '-DCROCODDYL_SOURCE_DIR=\\\"${${PROJECT_NAME}_SOURCE_DIR}\\\"')
endforeach(NAME ${${PROJECT_NAME}_CPP_TESTS})

if(BUILD_WITH_CODEGEN_SUPPORT)
  foreach(TEST_NAME ${${PROJECT_NAME}_CODEGEN_CPP_TESTS})
    target_link_libraries(${TEST_NAME} ${CMAKE_DL_LIBS} ${cppad_LIBRARY})
  endforeach()
endif()
