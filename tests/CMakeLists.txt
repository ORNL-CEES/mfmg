INCLUDE(${CMAKE_SOURCE_DIR}/cmake/SetupTests.cmake)

# Link against the Unit Test Framework of Boost
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

# Tests to run. Tests can be run using x, y, and z processors using
# MFMG_ADD_TEST(test x y z)
MFMG_ADD_TEST(test_laplace 1 2 4)
