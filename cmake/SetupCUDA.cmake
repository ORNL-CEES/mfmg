# This function set the macro MFMG_WITH_CUDA and set MFMG_CUDA_LIBRARIES with
# the list of CUDA libraries that we are using
FUNCTION(SET_CUDA_LIBRARIES)
  ADD_DEFINITIONS(-DMFMG_WITH_CUDA)
  FIND_LIBRARY(CUSPARSE cusparse PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  FIND_LIBRARY(CUSOLVER cusolver PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  MESSAGE("implicit ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
  SET(MFMG_CUDA_LIBRARIES
    ${CUSPARSE}
    ${CUSOLVER}
    PARENT_SCOPE
    )

#### OPENMP ##################################################################
# cuSOLVER needs OpenMP
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}"
      PARENT_SCOPE
      )
  ELSE()
    MESSAGE(SEND_ERROR "Could not find OpenMP required by cuSolver")
  ENDIF()

#### AMGX ####################################################################
  IF(${MFMG_ENABLE_CUDA})
    IF (${MFMG_ENABLE_AMGX})
      FIND_LIBRARY(AMGX_LIBRARY amgx amgxsh PATH "${AMGX_DIR}/lib")
      FIND_LIBRARY(CUBLAS cublas HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
      SET(AMGX_LIBRARIES
        ${AMGX_LIBRARY}
        ${CUBLAS}
        PARENT_SCOPE
        )
      SET(AMGX_INCLUDE_DIR "${AMGX_DIR}/include" PARENT_SCOPE)
      ADD_DEFINITIONS(-DMFMG_WITH_AMGX)
    ENDIF()
  ENDIF()
ENDFUNCTION()
