SET(${PROJECT_NAME}_EXAMPLES_PYTHON
  double_pendulum
  quadrotor
  quadrotor_ubound
  arm_manipulation
  quadrupedal_gaits
  quadrupedal_walk_ubound
  bipedal_walk
  bipedal_walk_ubound
  humanoid_manipulation
  humanoid_manipulation_ubound
  )

FOREACH(examples ${${PROJECT_NAME}_EXAMPLES_PYTHON})
  PYTHON_BUILD(. "${examples}.py")
  ADD_CUSTOM_TARGET("examples-${examples}"
    ${CMAKE_COMMAND} -E env PYTHONPATH=$ENV{PYTHONPATH}:${PROJECT_BINARY_DIR}/bindings/python
    ${PYTHON_EXECUTABLE} -c "import ${examples}" \${INPUT})
ENDFOREACH(examples ${${PROJECT_NAME}_EXAMPLES_PYTHON})
