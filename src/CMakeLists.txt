SET(${PROJECT_NAME}_SOURCES
  core/state-base.cpp
  core/action-base.cpp
  core/diff-action-base.cpp
  core/solver-base.cpp
  core/utils/callbacks.cpp
  core/optctrl/shooting.cpp
  core/solvers/ddp.cpp
  core/states/state-euclidean.cpp
  core/states/state-num-diff.cpp
  core/actions/action-unicycle.cpp
  core/actions/action-lqr.cpp
  core/actions/diff-action-lqr.cpp
  core/actions/action-num-diff.cpp
  )

IF(UNIX)
  ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES})
  SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  PKG_CONFIG_USE_DEPENDENCY(${PROJECT_NAME} eigen3)

  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib)
  INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
          DESTINATION include
          FILES_MATCHING PATTERN "*.h*")
ENDIF(UNIX)