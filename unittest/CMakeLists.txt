IF(BUILD_PYTHON_INTERFACE)
  ADD_SUBDIRECTORY(bindings)
ENDIF(BUILD_PYTHON_INTERFACE)


ADD_SUBDIRECTORY(python)

SET(${PROJECT_NAME}_CPP_TESTS
  test_states
  # test_actuation
  test_activations
  test_actions
  # test_integrators
  test_diff_actions
  test_impulses
  test_multiple_impulses
  test_friction_cone
  test_costs
  # test_contacts
  test_boxqp
  test_solvers
)

INCLUDE_DIRECTORIES(.)

MACRO(ADD_TEST_CFLAGS target flag)
  SET_PROPERTY(TARGET ${target} APPEND_STRING PROPERTY COMPILE_FLAGS " ${flag}")
ENDMACRO(ADD_TEST_CFLAGS)

MACRO(ADD_CPP_UNIT_TEST NAME PKGS)
  set(unittest_name ${NAME})

  ADD_UNIT_TEST(${unittest_name} ${NAME}.cpp)

  SET_TARGET_PROPERTIES(${unittest_name} PROPERTIES LINKER_LANGUAGE CXX)

  ADD_TEST_CFLAGS(${unittest_name} "-DBOOST_TEST_DYN_LINK")

  SET(MODULE_NAME "${NAME}Test")
  STRING(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})
  ADD_TEST_CFLAGS(${unittest_name} "-DTEST_MODULE_NAME=${MODULE_NAME}")
  ADD_TEST_CFLAGS(${unittest_name} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

  FOREACH(PKG ${PKGS})
    PKG_CONFIG_USE_DEPENDENCY(${unittest_name} ${PKG})
  ENDFOREACH(PKG)

  TARGET_LINK_LIBRARIES(${unittest_name} ${PROJECT_NAME})
  TARGET_LINK_LIBRARIES(${unittest_name} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

  ADD_TEST_CFLAGS(${unittest_name} '-DCROCODDYL_SOURCE_DIR=\\\"${${PROJECT_NAME}_SOURCE_DIR}\\\"')
ENDMACRO(ADD_CPP_UNIT_TEST NAME PKGS)


FOREACH(TEST ${${PROJECT_NAME}_CPP_TESTS})
  ADD_CPP_UNIT_TEST(${TEST} "eigen3;pinocchio;example-robot-data")
ENDFOREACH(TEST ${${PROJECT_NAME}_CPP_TESTS})
