# Set the notebook file path
set(${PROJECT_NAME}_NOTEBOOK_TESTS 02_optimizing_a_cartpole_swingup)

# Append the codegen notebooks to the list of tests
if(BUILD_WITH_CODEGEN_SUPPORT)
  list(APPEND ${PROJECT_NAME}_NOTEBOOK_TESTS 05_codegenerating_a_cartpole)
endif()

# Add a custom test for running the notebook
foreach(NOTEBOOK ${${PROJECT_NAME}_NOTEBOOK_TESTS})
  python_build(. "run_notebook.py")
  add_custom_target(
    "notebook-${NOTEBOOK}"
    ${CMAKE_COMMAND}
    -E
    env
    PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
    ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/run_notebook.py
    ${CMAKE_CURRENT_SOURCE_DIR}/${NOTEBOOK}.ipynb)
  # Add a test for the notebook
  add_test(
    NAME "notebook-${NOTEBOOK}"
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_notebook.py
            ${CMAKE_CURRENT_SOURCE_DIR}/${NOTEBOOK}.ipynb
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  # Set the Python path for the test
  set_tests_properties(
    "notebook-${NOTEBOOK}"
    PROPERTIES
      ENVIRONMENT
      "PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}")
  install(FILES ${NOTEBOOK}.ipynb
          DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/notebooks)
endforeach(NOTEBOOK ${${PROJECT_NAME}_NOTEBOOK_TESTS})
