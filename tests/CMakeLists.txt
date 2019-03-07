INCLUDE(${CMAKE_SOURCE_DIR}/cmake/SetupTests.cmake)

# Link against the Unit Test Framework of Boost
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

# Tests to run. Tests can be run using x, y, and z processors using
# MFMG_ADD_TEST(test x y z)
MFMG_ADD_TEST(test_hierarchy 1 2 4)

MFMG_COPY_INPUT_FILE(hierarchy_input.info  tests/data)
MFMG_COPY_INPUT_FILE(amgx_config_fgmres.json  tests/data)
MFMG_COPY_INPUT_FILE(amgx_config_amg.json  tests/data)
