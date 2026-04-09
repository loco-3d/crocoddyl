if(BUILD_PYTHON_INTERFACE)
    set(${PROJECT_NAME}_PYTHON_TESTS
        test_collisions
        test_integration
        test_velocity_avoidance
    )

    foreach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
        set(TEST_NAME "${PROJECT_NAME}-test-py-${TEST}")
        add_python_unit_test(${TEST_NAME} "tests/${TEST}.py" "python")
    endforeach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
endif()
