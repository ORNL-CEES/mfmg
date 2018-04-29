IF(NOT CLANG_FORMAT_EXECUTABLE)
  FIND_PROGRAM(CLANG_FORMAT_EXECUTABLE
    NAMES
    clang-format-5.0
    clang-format-mp-5.0
    clang-format
    )
  IF(CLANG_FORMAT_EXECUTABLE)
    MESSAGE("-- Found clang-format: ${CLANG_FORMAT_EXECUTABLE}")
  ELSE()
    MESSAGE(SEND_ERROR "-- clang-format not found")
  ENDIF()
ELSE()
  MESSAGE("-- Using clang-format: ${CLANG_FORMAT_EXECUTABLE}")
  IF(NOT EXISTS ${CLANG_FORMAT_EXECUTABLE})
    MESSAGE(SEND_ERROR "-- clang-format path is invalid")
  ENDIF()
ENDIF()

# Check that the version of clang-format is the correct one
EXECUTE_PROCESS(
  COMMAND ${CLANG_FORMAT_EXECUTABLE} -version
  OUTPUT_VARIABLE CLANG_FORMAT_VERSION
  )
IF(NOT CLANG_FORMAT_VERSION MATCHES "5.0")
    MESSAGE(SEND_ERROR "You must use clang-format version 5.0")
ENDIF()

# Download diff-clang-format.py from ORNL-CEES/Cap
FILE(DOWNLOAD
  https://raw.githubusercontent.com/ORNL-CEES/Cap/master/diff-clang-format.py
  ${CMAKE_BINARY_DIR}/diff-clang-format.py
  STATUS status
  )
LIST(GET status 0 error_code)
IF(error_code)
  LIST(GET status 1 error_string)
  MESSAGE(WARNING "Failed downloading diff-clang-format.py from GitHub"
    " (${error_string})")
  MESSAGE("-- " "NOTE: Disabling C++ code formatting because "
    "diff-clang-format.py is missing")
  SET(skip TRUE)
ENDIF()

# Do not bother continuing if not able to fetch diff-clang-format.py
IF(NOT skip)
  FIND_PACKAGE(PythonInterp REQUIRED)

  # Download docopt command line argument parser
  FILE(DOWNLOAD
    https://raw.githubusercontent.com/docopt/docopt/0.6.2/docopt.py
    ${CMAKE_BINARY_DIR}/docopt.py
    )
  # Add a custom target that applies the C++ code formatting style to the source
  ADD_CUSTOM_TARGET(format
    ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/diff-clang-format.py
    --file-extension='.hpp'
    --file-extension='.cuh'
    --file-extension='.cc'
    --file-extension='.cu'
    --binary=${CLANG_FORMAT_EXECUTABLE}
    --style=file
    --config=${CMAKE_SOURCE_DIR}/.clang-format
    --apply-patch
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/source
    ${CMAKE_SOURCE_DIR}/tests
    )

  # Add a test that checks the code is formatted properly
  FILE(WRITE
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/check_format.sh
    "#!/usr/bin/env bash\n"
    "\n"
    "${PYTHON_EXECUTABLE} "
    "${CMAKE_BINARY_DIR}/diff-clang-format.py "
    "--file-extension='.hpp' --file-extension='.cuh' --file-extension='.cc' --file-extension='.cu' "
    "--binary=${CLANG_FORMAT_EXECUTABLE} "
    "--style=file "
    "--config=${CMAKE_SOURCE_DIR}/.clang-format "
    "${CMAKE_SOURCE_DIR}/include "
    "${CMAKE_SOURCE_DIR}/source "
    "${CMAKE_SOURCE_DIR}/tests "
    )
  FILE(COPY
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/check_format.sh
    DESTINATION
    ${CMAKE_BINARY_DIR}
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    )
  ADD_TEST(
    NAME check_format
    COMMAND ${CMAKE_BINARY_DIR}/check_format.sh
    )
ENDIF() # skip when download fails
