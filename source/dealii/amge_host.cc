/*************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#include <mfmg/dealii/amge_host.templates.hpp>
#include <mfmg/dealii/dealii_matrix_free_mesh_evaluator.hpp>
#include <mfmg/dealii/dealii_mesh_evaluator.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/lac/la_parallel_vector.h>

template class mfmg::AMGe_host<
    2, mfmg::DealIIMeshEvaluator<2>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::AMGe_host<
    3, mfmg::DealIIMeshEvaluator<3>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::AMGe_host<
    2, mfmg::DealIIMatrixFreeMeshEvaluator<2>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::AMGe_host<
    3, mfmg::DealIIMatrixFreeMeshEvaluator<3>,
    dealii::LinearAlgebra::distributed::Vector<double>>;

#define INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS(DIM, MESH_EVALUATOR)            \
  template std::tuple<                                                         \
      std::vector<std::complex<double>>, std::vector<dealii::Vector<double>>,  \
      std::vector<double>, std::vector<dealii::types::global_dof_index>>       \
  mfmg::AMGe_host<DIM, MESH_EVALUATOR<DIM>,                                    \
                  dealii::LinearAlgebra::distributed::Vector<double>>::        \
      compute_local_eigenvectors(                                              \
          unsigned int n_eigenvectors, double tolerance,                       \
          dealii::parallel::distributed::Triangulation<DIM> const              \
              &agglomerate_triangulation,                                      \
          std::map<                                                            \
              typename dealii::Triangulation<DIM>::active_cell_iterator,       \
              typename dealii::DoFHandler<DIM>::active_cell_iterator> const    \
              &patch_to_global_map,                                            \
          MESH_EVALUATOR<DIM> const &evaluator, int) const;

INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS(2, mfmg::DealIIMeshEvaluator)
INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS(3, mfmg::DealIIMeshEvaluator)
INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS(2, mfmg::DealIIMatrixFreeMeshEvaluator)
INSTANTIATE_COMPUTE_LOCAL_EIGENVECTORS(3, mfmg::DealIIMatrixFreeMeshEvaluator)
