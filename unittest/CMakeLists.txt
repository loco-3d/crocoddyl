ADD_SUBDIRECTORY(python)

SET(${PROJECT_NAME}_CPP_TESTS
  # test_actions.cpp
  # test_activation.cpp
  # test_actuation.cpp
  # test_armature.cpp
  # test_boxsolvers.cpp
  # test_contacts.cpp
  # test_costs.cpp
  # test_ddp_contact.cpp
  # test_dse3.cpp
  # test_dynamic_derivatives.cpp
  # test_impact.cpp
  # test_locomotion.cpp
  # test_quadruped.cpp
  # test_rk4.cpp
  # test_robots.cpp
  # test_solvers.cpp
  test_state.cpp
  )

MACRO(ADD_TEST_CFLAGS target flag)
  SET_PROPERTY(TARGET ${target} APPEND_STRING PROPERTY COMPILE_FLAGS " ${flag}")
ENDMACRO(ADD_TEST_CFLAGS)

MACRO(ADD_CPP_UNIT_TEST NAME PKGS)
  ADD_UNIT_TEST(${NAME} ${NAME})

  SET_TARGET_PROPERTIES(${NAME} PROPERTIES LINKER_LANGUAGE CXX)
  ADD_TEST_CFLAGS(${NAME} "-DBOOST_TEST_DYN_LINK")
  SET(MODULE_NAME "${NAME}Test")
  STRING(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})
  ADD_TEST_CFLAGS(${NAME} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

  FOREACH(PKG ${PKGS})
    PKG_CONFIG_USE_DEPENDENCY(${NAME} ${PKG})
  ENDFOREACH(PKG)

  TARGET_LINK_LIBRARIES(${NAME} ${PROJECT_NAME})
  TARGET_LINK_LIBRARIES(${NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
ENDMACRO(ADD_CPP_UNIT_TEST NAME PKGS)


FOREACH(TEST ${${PROJECT_NAME}_CPP_TESTS})
  ADD_CPP_UNIT_TEST(${TEST} "eigen3;pinocchio")
  ADD_TEST_CFLAGS(${TEST} '-DCROCODDYL_SOURCE_DIR=\\\"${${PROJECT_NAME}_SOURCE_DIR}\\\"')
ENDFOREACH(TEST ${${PROJECT_NAME}_CPP_TESTS})


ADD_SUBDIRECTORY(bindings)