#### Message Passing Interface (MPI) #########################################
FIND_PACKAGE(MPI REQUIRED)

#### Boost ###################################################################
IF(DEFINED BOOST_DIR)
  SET(BOOST_ROOT ${BOOST_DIR})
ENDIF()
SET(Boost_COMPONENTS
  unit_test_framework
  program_options
  )
FIND_PACKAGE(Boost 1.65.1 REQUIRED COMPONENTS ${Boost_COMPONENTS})

#### deal.II #################################################################
FIND_PACKAGE(deal.II 9.0 REQUIRED PATHS ${DEAL_II_DIR})

IF(NOT DEAL_II_WITH_TRILINOS)
  MESSAGE(FATAL_ERROR 
          "Error! deal.II must be compiled with Trilinos support.")
ENDIF()

INCLUDE(CheckCXXSourceCompiles)
SET(CMAKE_REQUIRED_INCLUDES ${DEAL_II_INCLUDE_DIRS})
CHECK_CXX_SOURCE_COMPILES(
  "
  #include <Anasazi_config.h>

  #ifdef ANASAZI_TEUCHOS_TIME_MONITOR
  error
  #endif
  int main(){}
  "
  MFMG_APPROPRIATE_ANASAZI
  )
UNSET(CMAKE_REQUIRED_INCLUDES)
IF(NOT MFMG_APPROPRIATE_ANASAZI)
  MESSAGE(FATAL_ERROR
          "Error! Trilinos was not configured with Anasazi support or defines "
	  "ANASAZI_TEUCHOS_TIME_MONITOR.")
ENDIF()

# If deal.II was configured in DebugRelease mode, then if mfmg was configured
# in Debug mode, we link against the Debug version of deal.II. If mfmg was
# configured in Release mode, we link against the Release version of deal.II. If
# we configure with the Debug version of deal.II, we need to use -DDEBUG to
# enable dealii::Assert in header files.
STRING(FIND "${DEAL_II_LIBRARIES}" "general" SINGLE_DEAL_II)
IF (${SINGLE_DEAL_II} EQUAL -1)
  IF(CMAKE_BUILD_TYPE MATCHES "Release")
    SET(DEAL_II_LIBRARIES ${DEAL_II_LIBRARIES_RELEASE})
  ELSE()
    SET(DEAL_II_LIBRARIES ${DEAL_II_LIBRARIES_DEBUG})
    ADD_DEFINITIONS(-DDEBUG)
  ENDIF()
ELSE()
  IF (NOT "${DEAL_II_TARGET_DEBUG}" STREQUAL "")
    ADD_DEFINITIONS(-DDEBUG)
  ENDIF()
ENDIF()

#### LAPACKE #################################################################
FIND_PATH(LAPACKE_INCLUDE_DIR lapacke.h
          HINTS ${LAPACK_DIR} $ENV{LAPACK_DIR}
          PATH_SUFFIXES include)

IF(NOT LAPACKE_INCLUDE_DIR)
  MESSAGE(STATUS "${LAPACKE_INCLUDE_DIR}")
  MESSAGE(FATAL_ERROR
          "Error! Could not locate LAPACKE.")
ENDIF()

# First test if the libraries deal.II links to already include LAPACKE
INCLUDE(CheckFunctionExists)
SET(CMAKE_REQUIRED_LIBRARIES "${DEAL_II_LIBRARIES}")
CHECK_FUNCTION_EXISTS(LAPACKE_dgeqrf LAPACKE_WORKS)

# If not try to find a separate library
IF (LAPACKE_WORKS)
  SET(LAPACKE_LIBRARY "")
ELSE()
  FIND_LIBRARY(LAPACKE_LIBRARY NAMES lapacke
               HINTS ${LAPACK_DIR} $ENV{LAPACK_DIR}
               PATH_SUFFIXES lib)

  IF(NOT LAPACKE_LIBRARY)
    MESSAGE(FATAL_ERROR
            "Error! Could not locate LAPACKE.")
  ENDIF()
ENDIF()

#### AMGX ####################################################################
IF(${MFMG_ENABLE_CUDA})
  IF (${MFMG_ENABLE_AMGX})
    ADD_LIBRARY(amgx STATIC IMPORTED)
    SET_TARGET_PROPERTIES(amgx PROPERTIES IMPORTED_LOCATION
      "${AMGX_DIR}/lib/libamgx.a"
      IMPORTED_LINK_INTERFACE_LIBRARIES "cublas")
    ADD_DEFINITIONS(-DMFMG_WITH_AMGX)
    SET(AMGX_INCLUDE_DIR "${AMGX_DIR}/include")
    SET(AMGX_LIBRARY amgx)
  ENDIF()
ENDIF()
