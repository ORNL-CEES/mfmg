# This function set the macro MFMG_WITH_CUDA and set MFMG_CUDA_LIBRARIES with
# the list of CUDA libraries that we are using
FUNCTION(SET_CUDA_LIBRARIES)
  ADD_DEFINITIONS(-DMFMG_WITH_CUDA)
  SET(MFMG_CUDA_LIBRARIES
    "cusparse"
    "cusolver"
    PARENT_SCOPE
    )
ENDFUNCTION()
