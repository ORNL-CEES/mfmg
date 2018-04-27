IF (NOT CLANG_TIDY_EXECUTABLE)
  FIND_PROGRAM(CLANG_TIDY_EXECUTABLE
    NAMES
    clang-tidy-5.0
    clang-tidy-mp-5.0
    clang-tidy
  )
  IF (CLANG_TIDY_EXECUTABLE)
    MESSAGE("-- Found clang-tidy " ${CLANG_TIDY_EXECUTABLE})
  ELSE()
    MESSAGE(SEND_ERROR "-- clang-tidy not found")
  ENDIF()
ELSE()
  MESSAGE("-- Using clang-tidy: ${CLANG_TIDY_EXECUTABLE}")
  IF(NOT EXISTS ${CLANG_TIDY_EXECUTABLE})
    MESSAGE(SEND_ERROR "-- clang-tidy path is invalid")
  ENDIF()
ENDIF()

# Check that the version of clang-format is the correct one
EXECUTE_PROCESS(
  COMMAND ${CLANG_TIDY_EXECUTABLE} -version
  OUTPUT_VARIABLE CLANG_TIDY_VERSION
  )
IF(NOT CLANG_TIDY_VERSION MATCHES "5.0")
    MESSAGE(SEND_ERROR "You must use clang-tidy version 5.0")
ENDIF()

# Run clang-tidy on each of the C++ source file of the project
SET(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")

# Create a preprocessor definition that depends on .clang-tidy content so
# the compile command will change when .clang-tidy changes.  This ensures
# that a subsequent build re-runs clang-tidy on all sources even if they
# do not otherwise need to be recompiled.  Nothing actually uses this
# definition.  We add it to targets on which we run clang-tidy just to
# get the build dependency on the .clang-tidy file.
FILE(SHA1 ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
SET(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
UNSET(clang_tidy_sha1)

CONFIGURE_FILE(.clang-tidy .clang-tidy COPYONLY)
