file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_BINDINGS_HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_BINDINGS_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(${PROJECT_NAME}_PYTHON_BINDINGS_FILES __init__.py deprecated.py)

# Define the python installation directory
set(CROCODDYL_PYTHON_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})
set(STUBGEN_DEPENDENCIES "${PROJECT_NAME}_pywrap_float64")

# Function to build Python bindings for a given scalar type
function(crocoddyl_python_library_for_scalar scalar)
  # Map scalar types to internal scalar names
  if("${scalar}" STREQUAL "double")
    set(SCALAR_NAME "float64")
  elseif("${scalar}" STREQUAL "float")
    set(SCALAR_NAME "float32")
  elseif("${scalar}" STREQUAL "CppAD::AD<CppAD::cg::CG<double>>")
    set(SCALAR_NAME "cgfloat64")
  else()
    message(FATAL_ERROR "Unknown scalar type: ${scalar}")
  endif()
  # Define the Boost.Python wrapper target name
  set(PYWRAP "${PROJECT_NAME}_pywrap_${SCALAR_NAME}")
  list(APPEND STUBGEN_DEPENDENCIES "${PYWRAP}")
  # Add the shared library for Boost.Python wrapper
  add_library(${PYWRAP} SHARED ${${PROJECT_NAME}_PYTHON_BINDINGS_SOURCES}
                               ${${PROJECT_NAME}_PYTHON_BINDINGS_HEADERS})
  set_target_properties(${PYWRAP} PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})
  # Link the target against the main library and EigenPy
  target_link_libraries(${PYWRAP} ${PROJECT_NAME} eigenpy::eigenpy)
  # Link against pycppad if code generation is enabled
  if(BUILD_WITH_CODEGEN_SUPPORT)
    target_link_libraries(${PYWRAP} ${PROJECT_NAME} pycppad::pycppad)
  endif()
  # Suppress specific compiler warnings for wrapper code
  # BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS spews conversion warnings from int to
  # long unsigned int. Unfortunately, using literals does not work in a macro.
  # As such, this turns them off for the entire wrapper:
  target_compile_options(${PYWRAP} PRIVATE "-Wno-conversion")
  # We are also exposing deprecated declarations until they are removed. Ignore
  # the errors for the wrapper:
  target_compile_options(${PYWRAP} PRIVATE "-Wno-deprecated-declarations")
  # Ensure .pyc files are built before generating stubs
  python_build_get_target(python_build_target)
  add_dependencies(${PYWRAP} ${python_build_target})
  # Set install RPATH on UNIX systems
  if(UNIX)
    get_relative_rpath(${CROCODDYL_PYTHON_INSTALL_DIR} ${PYWRAP}_INSTALL_RPATH)
    set_target_properties(${PYWRAP} PROPERTIES INSTALL_RPATH
                                               "${${PYWRAP}_INSTALL_RPATH}")
  endif()
  # Determine if the scalar type is a floating-point type
  list(FIND SCALAR_FPTYPES "${scalar}" scalar_index)
  if(NOT scalar_index EQUAL -1)
    set(FP_TYPE TRUE)
  else()
    set(FP_TYPE FALSE)
  endif()
  # Generate the scalar-specific C++ files
  crocoddyl_generate_cpp_files(${PYWRAP} cpp bindings/python ${scalar})
  # Install the Boost.Python wrapper target
  install(TARGETS ${PYWRAP} DESTINATION ${CROCODDYL_PYTHON_INSTALL_DIR})
  # Create and install a scalar-specific __init__.py if not float64
  if(NOT "${SCALAR_NAME}" STREQUAL "float64")
    set(SCALAR_TYPE ${SCALAR_NAME})
    configure_file(__init__.py.in
                   ${CMAKE_CURRENT_BINARY_DIR}/${SCALAR_NAME}/__init__.py @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${SCALAR_NAME}/__init__.py
            DESTINATION ${CROCODDYL_PYTHON_INSTALL_DIR}/${SCALAR_NAME})
  endif()
endfunction()

# Build the Python wrapper for each scalar type
foreach(scalar ${SCALAR_TYPES})
  crocoddyl_python_library_for_scalar(${scalar})
endforeach()

# Generate Python type stubs if enabled
if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(
    ${CMAKE_CURRENT_BINARY_DIR}/.. ${PROJECT_NAME} ${PYTHON_SITELIB}
    ${PYWRAP}_float64 ${PROJECT_NAME}_float64 ${STUBGEN_DEPENDENCIES})
endif()

# Install the main Python binding files
foreach(python ${${PROJECT_NAME}_PYTHON_BINDINGS_FILES})
  python_build(. ${python})
  install(
    FILES "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/crocoddyl/${python}"
    DESTINATION ${CROCODDYL_PYTHON_INSTALL_DIR})
endforeach(python ${${PROJECT_NAME}_PYTHON_BINDINGS_FILES})
# Install static files for Python interface
install(
  FILES
    "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/crocoddyl/crocoddyl.launch"
    "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/crocoddyl/crocoddyl.rviz"
  DESTINATION ${CROCODDYL_PYTHON_INSTALL_DIR})
# Install utility Python files
set(${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES __init__.py pendulum.py
                                                biped.py quadruped.py)

foreach(python ${${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES})
  python_build(utils ${python})
  install(
    FILES
      "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/crocoddyl/utils/${python}"
    DESTINATION ${CROCODDYL_PYTHON_INSTALL_DIR}/utils)
endforeach(python ${${PROJECT_NAME}_BINDINGS_UTILS_PYTHON_FILES})
