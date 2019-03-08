SET(${PROJECT_NAME}_PYTHON_FILES
  action.py
  activation.py
  actuation.py
  callbacks.py
  contact.py
  cost.py
  ddp.py
  diagnostic.py
  differential_action.py
  fddp.py
  floating_contact.py
  impact.py
  __init__.py
  integrated_action.py
  kkt.py
  robots.py
  shooting.py
  state.py
  unicycle.py
  utils.py
  )

FOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
  PYTHON_BUILD(. ${python})
  INSTALL(FILES ${python} DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
ENDFOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
