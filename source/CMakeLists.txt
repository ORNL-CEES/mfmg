SET(MFMG_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/amge.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/amge_host.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/dealii_operator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  )

IF (${MFMG_ENABLE_CUDA})
  SET(MFMG_SOURCES
    ${MFMG_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/amge_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/dealii_operator_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/dealii_operator_device_helpers.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_matrix_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cu
    )
ENDIF()

SET(MFMG_SOURCES ${MFMG_SOURCES} PARENT_SCOPE)
