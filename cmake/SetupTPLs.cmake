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
FIND_PACKAGE(deal.II 9.0 REQUIRED PATHS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

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

#### AMGX ####################################################################
IF(${MFMG_ENABLE_CUDA})
  IF (${MFMG_ENABLE_AMGX})
    FIND_LIBRARY(AMGX_LIBRARY NAMES amgx
      HINTS ${AMGX_DIR} $ENV{AMGX_DIR}
      PATH_SUFFIXES lib)
    FIND_PATH(AMGX_INCLUDE_DIR amgx_c.h
      HINTS ${AMGX_DIR} $ENV{AMGX_DIR}
      PATH_SUFFIXES include)

    IF(NOT AMGX_INCLUDE_DIR OR NOT AMGX_LIBRARY)
      MESSAGE(FATAL_ERROR 
              "Error! Could not locate AmgX.")
    ENDIF()

    ADD_LIBRARY(amgx STATIC IMPORTED)
    SET_TARGET_PROPERTIES(amgx PROPERTIES IMPORTED_LOCATION
      ${AMGX_LIBRARY}
      IMPORTED_LINK_INTERFACE_LIBRARIES "cublas")
    ADD_DEFINITIONS(-DMFMG_WITH_AMGX)
    SET(AMGX_LIBRARY amgx)
  ENDIF()
ENDIF()
