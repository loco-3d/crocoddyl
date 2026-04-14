set(PROJECT_NAME colmpc)
set(PY_NAME "${PROJECT_NAME}_pywrap")
set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB})

if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  set(${PROJECT_NAME}_HEADERS
      include/${PROJECT_NAME}/fwd.hpp
      include/${PROJECT_NAME}/residual-distance-collision.hpp
      include/${PROJECT_NAME}/residual-distance-collision.hxx
      include/${PROJECT_NAME}/residual-velocity-avoidance.hpp
      include/${PROJECT_NAME}/residual-velocity-avoidance.hxx
      include/${PROJECT_NAME}/activation-distance-quadratic.hpp
      include/${PROJECT_NAME}/quadratic-exp.hpp)

  install(
    FILES ${${PROJECT_NAME}_HEADERS}
    DESTINATION "include/${PROJECT_NAME}"
    PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)

  add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_HEADERS})
  add_library(crocoddyl::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(${PROJECT_NAME} PUBLIC crocoddyl::crocoddyl)
  if(UNIX)
    get_relative_rpath(${CMAKE_INSTALL_LIBDIR} ${PROJECT_NAME}_INSTALL_RPATH)
    set_target_properties(
      ${PROJECT_NAME} PROPERTIES INSTALL_RPATH
                                 "${${PROJECT_NAME}_INSTALL_RPATH}")
  endif()

  if(SUFFIX_SO_VERSION)
    set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION
                                                     ${PROJECT_VERSION})
  endif()
endif()

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

# # Unit tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Installation
if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
