INCLUDE(${CMAKE_SOURCE_DIR}/cmake/SetupTests.cmake)

# Link against the Unit Test Framework of Boost
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

# Tests to run. Tests can be run using x, y, and z processors using
# MFMG_ADD_TEST(test x y z)
MFMG_ADD_TEST(test_laplace 1 2 4)
MFMG_ADD_TEST(test_agglomerate 1 2 4)
MFMG_ADD_TEST(test_eigenvectors 1)
MFMG_ADD_TEST(test_restriction_matrix 1 2 4)

IF(${MFMG_ENABLE_CUDA})
  MFMG_ADD_CUDA_TEST(test_utils 1 2 4)
  MFMG_ADD_CUDA_TEST(test_eigenvectors_device 1)
  MFMG_ADD_CUDA_TEST(test_restriction_matrix_device 1)
ENDIF()
