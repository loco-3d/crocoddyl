
SET(${PROJECT_NAME}_HEADERS
  crocoddyl/core/state-base.hpp
  crocoddyl/core/action-base.hpp
  crocoddyl/core/solver-base.hpp
  crocoddyl/core/optctrl/shooting.hpp
  crocoddyl/core/solvers/ddp.hpp
  crocoddyl/core/actions/unicycle.hpp
  crocoddyl/core/actions/lqr.hpp
  crocoddyl/core/states/state-euclidean.hpp
  crocoddyl/core/utils/callbacks.hpp
  )

IF(UNIX)
  ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_HEADERS})
  SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  PKG_CONFIG_USE_DEPENDENCY(${PROJECT_NAME} eigen3)

  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib)
  INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
          DESTINATION include
          FILES_MATCHING PATTERN "*.h*")
ENDIF(UNIX)



#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
ADD_EXECUTABLE(tmp  crocoddyl/tmp.cpp)
TARGET_LINK_LIBRARIES(tmp ${PROJECT_NAME})
