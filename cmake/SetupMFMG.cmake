# Set flags for Release and Debug version
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")

# Disable boost asserts in Release mode and set MFMG_DEBUG in Debug mode
MESSAGE("-- Build type: ${CMAKE_BUILD_TYPE}")
IF(CMAKE_BUILD_TYPE MATCHES "Release" OR CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  # Do nothing
ELSEIF(CMAKE_BUILD_TYPE MATCHES "Debug")
  ADD_DEFINITIONS(-DMFMG_DEBUG)
ELSE()
  MESSAGE(SEND_ERROR
    "Possible values for CMAKE_BUILD_TYPE are Release, RelWithDebInfo, and Debug"
    )
ENDIF()
