#### Message Passing Interface (MPI) #########################################
FIND_PACKAGE(MPI REQUIRED)

#### Boost ###################################################################
IF(DEFINED BOOST_DIR)
  SET(BOOST_ROOT ${BOOST_DIR})
ENDIF()
SET(Boost_COMPONENTS
  mpi
  unit_test_framework
  )
FIND_PACKAGE(Boost 1.65.1 REQUIRED COMPONENTS ${Boost_COMPONENTS})

#### deal.II #################################################################
FIND_PACKAGE(deal.II 8.5 REQUIRED PATHS ${DEAL_II_DIR})

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
