SET(MFMG_SOURCES
  ${MFMG_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/amge_device.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_handle.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_hierarchy_helpers.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_matrix_operator.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_matrix_free_operator.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mesh_evaluator.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_matrix_free_mesh_evaluator.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_smoother.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_solver.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_operator_device_helpers.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_matrix_device.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cu
  )

SET(MFMG_SOURCES ${MFMG_SOURCES} PARENT_SCOPE)
