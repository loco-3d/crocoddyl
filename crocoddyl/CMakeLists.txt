SET(${PROJECT_NAME}_PYTHON_FILES
  action.py
  activation.py
  actuation.py
  box_kkt.py
  callbacks.py
  contact.py
  cost.py
  ddp.py
  box_ddp.py
  diagnostic.py
  differential_action.py
  fddp.py
  floating_contact.py
  impact.py
  __init__.py
  integrated_action.py
  qpsolvers.py
  kkt.py
  robots.py
  shooting.py
  solver.py
  state.py
  unicycle.py
  utils.py
  )

FOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
  PYTHON_BUILD(. ${python})
  INSTALL(FILES ${python} DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
ENDFOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})

SET(${PROJECT_NAME}_LOCOMOTION_PYTHON_FILES
  __init__.py
  centroidal_phi.py
  contact_sequence_wrapper.py
  problem_definition.py
  spline_utils.py
  )

FOREACH(python ${${PROJECT_NAME}_LOCOMOTION_PYTHON_FILES})
  PYTHON_BUILD(locomotion ${python})
  INSTALL(FILES ${python} DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/locomotion)
ENDFOREACH(python ${${PROJECT_NAME}_LOCOMOTION_PYTHON_FILES})
