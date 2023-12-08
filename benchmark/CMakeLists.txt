set(${PROJECT_NAME}_BENCHMARK
    boxqp
    unicycle_optctrl
    lqr_optctrl
    arm_manipulation_optctrl
    quadrupedal_gaits_optctrl
    arm_manipulation_timings
    bipedal_timings)

set(${PROJECT_NAME}_CODEGEN_BENCHMARK all_robots)
list(APPEND ${PROJECT_NAME}_BENCHMARK ${${PROJECT_NAME}_CODEGEN_BENCHMARK})

foreach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK})
  add_executable(${BENCHMARK_NAME} ${BENCHMARK_NAME}.cpp)
  target_link_libraries(${BENCHMARK_NAME} ${PROJECT_NAME}
                        example-robot-data::example-robot-data)
  add_custom_target("benchmarks-cpp-${BENCHMARK_NAME}" ${BENCHMARK_NAME}
                                                       \${INPUT})
endforeach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK})

if(BUILD_WITH_CODEGEN_SUPPORT)
  foreach(BENCHMARK_NAME ${${PROJECT_NAME}_CODEGEN_BENCHMARK})
    target_link_libraries(${BENCHMARK_NAME} ${CMAKE_DL_LIBS} ${cppad_LIBRARY})
  endforeach(BENCHMARK_NAME ${${PROJECT_NAME}_CODEGEN_BENCHMARK})
endif()

if(BUILD_PYTHON_INTERFACE)
  set(${PROJECT_NAME}_BENCHMARK_PYTHON
      unicycle_optctrl lqr_optctrl arm_manipulation_optctrl
      bipedal_walk_optctrl quadrupedal_gaits_optctrl)

  foreach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK_PYTHON})
    python_build(. "${BENCHMARK_NAME}.py")
    add_custom_target(
      "benchmarks-py-${BENCHMARK_NAME}"
      ${CMAKE_COMMAND}
      -E
      env
      PYTHONPATH=${PROJECT_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}
      ${PYTHON_EXECUTABLE}
      -c
      "from ${BENCHMARK_NAME} import *"
      \${INPUT})
  endforeach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK_PYTHON})
endif()
