if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python/crocoddyl)
  set(PYWRAP
      ${PYWRAP}
      PARENT_SCOPE)
endif()
