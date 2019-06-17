ADD_SUBDIRECTORY(python)

SET(${PROJECT_NAME}_CPP_TESTS
  # test_actions
  # test_activation
  # test_actuation
  # test_armature
  # test_boxsolvers
  # test_contacts
  # test_costs
  # test_ddp_contact
  # test_dse3
  # test_dynamic_derivatives
  # test_impact
  # test_locomotion
  # test_quadruped
  # test_rk4
  # test_robots
  # test_solvers
  test_state
  )

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
  ADD_TEST_CFLAGS(${unittest_name} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

  FOREACH(PKG ${PKGS})
    PKG_CONFIG_USE_DEPENDENCY(${unittest_name} ${PKG})
  ENDFOREACH(PKG)

  TARGET_LINK_LIBRARIES(${unittest_name} ${PROJECT_NAME})
  MESSAGE(WARNING "Boost unnittest libs are: ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}")
  TARGET_LINK_LIBRARIES(${unittest_name} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDMACRO(ADD_CPP_UNIT_TEST NAME PKGS)


FOREACH(TEST ${${PROJECT_NAME}_CPP_TESTS})
  ADD_CPP_UNIT_TEST(${TEST} "eigen3;pinocchio")
  ADD_TEST_CFLAGS(${TEST} '-DCROCODDYL_SOURCE_DIR=\\\"${${PROJECT_NAME}_SOURCE_DIR}\\\"')
ENDFOREACH(TEST ${${PROJECT_NAME}_CPP_TESTS})


ADD_SUBDIRECTORY(bindings)