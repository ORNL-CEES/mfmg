/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 *************************************************************************/

#include <mfmg/cuda/amge_device.templates.cuh>
#include <mfmg/cuda/cuda_mesh_evaluator.cuh>

#include <deal.II/lac/la_parallel_vector.h>

// Cannot use the instantiation macro with nvcc
template class mfmg::AMGe_device<2, mfmg::CudaMeshEvaluator<2>,
                                 dealii::LinearAlgebra::distributed::Vector<
                                     double, dealii::MemorySpace::Host>>;
template class mfmg::AMGe_device<3, mfmg::CudaMeshEvaluator<3>,
                                 dealii::LinearAlgebra::distributed::Vector<
                                     double, dealii::MemorySpace::Host>>;

template class mfmg::AMGe_device<2, mfmg::CudaMeshEvaluator<2>,
                                 dealii::LinearAlgebra::distributed::Vector<
                                     double, dealii::MemorySpace::CUDA>>;
template class mfmg::AMGe_device<3, mfmg::CudaMeshEvaluator<3>,
                                 dealii::LinearAlgebra::distributed::Vector<
                                     double, dealii::MemorySpace::CUDA>>;

template class mfmg::AMGe_device<2, mfmg::CudaMatrixFreeMeshEvaluator<2>,
                                 dealii::LinearAlgebra::distributed::Vector<
                                     double, dealii::MemorySpace::Host>>;
template class mfmg::AMGe_device<3, mfmg::CudaMatrixFreeMeshEvaluator<3>,
                                 dealii::LinearAlgebra::distributed::Vector<
                                     double, dealii::MemorySpace::Host>>;

template class mfmg::AMGe_device<2, mfmg::CudaMatrixFreeMeshEvaluator<2>,
                                 dealii::LinearAlgebra::distributed::Vector<
                                     double, dealii::MemorySpace::CUDA>>;
template class mfmg::AMGe_device<3, mfmg::CudaMatrixFreeMeshEvaluator<3>,
                                 dealii::LinearAlgebra::distributed::Vector<
                                     double, dealii::MemorySpace::CUDA>>;

#define INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS_CUDA(DIM, MESH_EVALUATOR)       \
  template std::tuple<double *, double *, double *, std::vector<unsigned int>> \
  mfmg::AMGe_device<DIM, MESH_EVALUATOR<DIM>,                                  \
                    dealii::LinearAlgebra::distributed::Vector<                \
                        double, dealii::MemorySpace::CUDA>>::                  \
      compute_local_eigenvectors(                                              \
          unsigned int n_eigenvectors, double tolerance,                       \
          dealii::parallel::distributed::Triangulation<DIM> const              \
              &agglomerate_triangulation,                                      \
          std::map<                                                            \
              typename dealii::Triangulation<DIM>::active_cell_iterator,       \
              typename dealii::DoFHandler<DIM>::active_cell_iterator> const    \
              &patch_to_global_map,                                            \
          MESH_EVALUATOR<DIM> const &evaluator, int);

INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS_CUDA(2, mfmg::CudaMeshEvaluator)
INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS_CUDA(3, mfmg::CudaMeshEvaluator)
INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS_CUDA(2,
                                            mfmg::CudaMatrixFreeMeshEvaluator)
INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS_CUDA(3,
                                            mfmg::CudaMatrixFreeMeshEvaluator)
