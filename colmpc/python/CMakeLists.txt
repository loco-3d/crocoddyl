set(${PY_NAME}_SOURCES
    main.cpp
    residual-distance-collision.cpp
    residual-velocity-avoidance.cpp
    activation-distance-quadratic.cpp
    quadratic-exp.cpp
    free-fwddyn.cpp
    state-multibody.cpp
    free-fwddyn-instantiation.cpp
    residual-distance-collision-2.cpp
)

add_library(${PY_NAME} SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(
    ${PY_NAME}
    PUBLIC ${PROJECT_NAME}::${PROJECT_NAME} eigenpy::eigenpy
)
set_target_properties(
    ${PY_NAME}
    PROPERTIES PREFIX "" OUTPUT_NAME ${PROJECT_NAME} SUFFIX ${PYTHON_EXT_SUFFIX}
)
if(UNIX)
    get_relative_rpath(${${PY_NAME}_INSTALL_DIR} ${PY_NAME}_INSTALL_RPATH)
    set_target_properties(
        ${PY_NAME}
        PROPERTIES INSTALL_RPATH "${${PY_NAME}_INSTALL_RPATH}"
    )
endif()
target_link_boost_python(${PY_NAME} PRIVATE)
install(TARGETS ${PY_NAME} DESTINATION ${${PY_NAME}_INSTALL_DIR})

if(GENERATE_PYTHON_STUBS)
    load_stubgen()
    generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_NAME} ${PYTHON_SITELIB}
                   ${PY_NAME}
    )
endif(GENERATE_PYTHON_STUBS)
