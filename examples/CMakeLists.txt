SET(${PROJECT_NAME}_EXAMPLES_PYTHON
  arm_manipulation
  bipedal_walk
  quadrupedal_gaits
  quadrupedal_walk_boxddp
  )

FOREACH(examples ${${PROJECT_NAME}_EXAMPLES_PYTHON})
  PYTHON_BUILD(. "${examples}.py")
  ADD_CUSTOM_TARGET("examples-${examples}"
    ${CMAKE_COMMAND} -E env PYTHONPATH=$ENV{PYTHONPATH}:${PROJECT_BINARY_DIR}/bindings/python
    ${PYTHON_EXECUTABLE} -c "import ${examples}")
ENDFOREACH(examples ${${PROJECT_NAME}_EXAMPLES_PYTHON})
