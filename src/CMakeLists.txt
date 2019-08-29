SET(${PROJECT_NAME}_SOURCES
  core/state-base.cpp
  core/actuation-base.cpp
  core/action-base.cpp
  core/diff-action-base.cpp
  core/activation-base.cpp
  core/solver-base.cpp
  core/integrator/euler.cpp
  core/numdiff/state.cpp
  core/numdiff/action.cpp
  core/numdiff/diff-action.cpp
  core/utils/callbacks.cpp
  core/optctrl/shooting.cpp
  core/solvers/ddp.cpp
  core/states/euclidean.cpp
  core/actions/unicycle.cpp
  core/actions/lqr.cpp
  core/actions/diff-lqr.cpp
  core/activations/quadratic.cpp
  core/activations/weighted-quadratic.cpp
  multibody/cost-base.cpp
  multibody/contact-base.cpp
  multibody/states/multibody.cpp
  multibody/actuations/floating-base.cpp
  multibody/actuations/full.cpp
  multibody/costs/cost-sum.cpp
  multibody/costs/state.cpp
  multibody/costs/control.cpp
  multibody/costs/com-position.cpp
  multibody/costs/frame-placement.cpp
  multibody/costs/frame-translation.cpp
  multibody/costs/frame-velocity.cpp
  multibody/contacts/multiple-contacts.cpp
  multibody/contacts/contact-3d.cpp
  multibody/contacts/contact-6d.cpp
  multibody/actions/free-fwddyn.cpp
  multibody/actions/contact-fwddyn.cpp
  )

IF(UNIX)
  ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES})
  SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  PKG_CONFIG_USE_DEPENDENCY(${PROJECT_NAME} eigen3)
  PKG_CONFIG_USE_DEPENDENCY(${PROJECT_NAME} pinocchio)
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_SERIALIZATION_LIBRARY})

  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib)
  INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
          DESTINATION include
          FILES_MATCHING PATTERN "*.h*")
ENDIF(UNIX)
