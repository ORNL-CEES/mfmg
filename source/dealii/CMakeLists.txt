SET(MFMG_SOURCES
  ${MFMG_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/amge_host.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_hierarchy_helpers.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_matrix_operator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_mesh_evaluator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_smoother.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_solver.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_trilinos_matrix_operator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_matrix_free_operator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_matrix_free_hierarchy_helpers.cc
  )

SET(MFMG_SOURCES ${MFMG_SOURCES} PARENT_SCOPE)
