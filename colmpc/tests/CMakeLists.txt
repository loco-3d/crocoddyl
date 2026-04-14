if(BUILD_PYTHON_INTERFACE)
  set(${PROJECT_NAME}_PYTHON_TESTS
      test_collisions
      # test_integration TODO: need numdifftools test_velocity_avoidance TODO:
      # need mim-solvers
  )

  foreach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
    set(TEST_NAME "${PROJECT_NAME}-test-py-${TEST}")
    add_python_unit_test(${TEST_NAME} "colmpc/tests/${TEST}.py"
                         "python;bindings/python")
  endforeach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
endif()
