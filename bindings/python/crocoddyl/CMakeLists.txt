file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_BINDINGS_HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_BINDINGS_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(${PROJECT_NAME}_PYTHON_BINDINGS_FILES __init__.py deprecated.py
                                          crocoddyl.launch crocoddyl.rviz)

set(PYWRAP "${PROJECT_NAME}_pywrap")
set(${PYWRAP}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})
add_library(${PYWRAP} SHARED ${${PROJECT_NAME}_PYTHON_BINDINGS_SOURCES}
                             ${${PROJECT_NAME}_PYTHON_BINDINGS_HEADERS})
set_target_properties(${PYWRAP} PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})

target_link_libraries(${PYWRAP} ${PROJECT_NAME} eigenpy::eigenpy)
# BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS spews conversion warnings from int to
# long unsigned int. Unfortunately, using literals does not work in a macro. As
# such, this turns them off for the entire wrapper:
target_compile_options(${PYWRAP} PRIVATE "-Wno-conversion")
# We are also exposing deprecated declarations until they are removed. Ignore
# the errors for the wrapper:
target_compile_options(${PYWRAP} PRIVATE "-Wno-deprecated-declarations")

# Enforce the build of the pyc before the stub generation (stub generation
# depend of PYWRAP)
python_build_get_target(python_build_target)
add_dependencies(${PYWRAP} ${python_build_target})

if(UNIX)
  get_relative_rpath(${${PYWRAP}_INSTALL_DIR} ${PYWRAP}_INSTALL_RPATH)
  set_target_properties(${PYWRAP} PROPERTIES INSTALL_RPATH
                                             "${${PYWRAP}_INSTALL_RPATH}")
endif()

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR}/.. ${PROJECT_NAME}
                 ${PYTHON_SITELIB} ${PYWRAP})
endif()

install(TARGETS ${PYWRAP} DESTINATION ${${PYWRAP}_INSTALL_DIR})

foreach(python ${${PROJECT_NAME}_PYTHON_BINDINGS_FILES})
  python_build(. ${python})
  install(
    FILES "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/crocoddyl/${python}"
    DESTINATION ${${PYWRAP}_INSTALL_DIR})
endforeach(python ${${PROJECT_NAME}_PYTHON_BINDINGS_FILES})

set(${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES __init__.py pendulum.py
                                                biped.py quadruped.py)

foreach(python ${${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES})
  python_build(utils ${python})
  install(
    FILES
      "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/crocoddyl/utils/${python}"
    DESTINATION ${${PYWRAP}_INSTALL_DIR}/utils)
endforeach(python ${${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES})
