SET(MFMG_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/amge.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/amge_host.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_hierarchy_helpers.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_matrix_operator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_mesh_evaluator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_smoother.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_solver.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_trilinos_matrix_operator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  )

IF (${MFMG_ENABLE_CUDA})
  SET(MFMG_SOURCES
    ${MFMG_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/amge_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_handle.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_hierarchy_helpers.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_matrix_operator.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mesh_evaluator.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_smoother.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_solver.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/dealii_operator_device_helpers.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_matrix_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_device.cu
    )
ENDIF()

SET(MFMG_SOURCES ${MFMG_SOURCES} PARENT_SCOPE)
