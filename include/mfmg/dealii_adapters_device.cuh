/*************************************************************************
 * Copyright (c) 2018 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#ifndef MFMG_DEALII_ADAPTERS_DEVICE_CUH
#define MFMG_DEALII_ADAPTERS_DEVICE_CUH

#include <mfmg/amge_device.cuh>
#include <mfmg/concepts.hpp>
#include <mfmg/dealii_mesh.hpp>
#include <mfmg/dealii_operator_device.cuh>

#include <boost/property_tree/ptree.hpp>

namespace mfmg
{
template <int dim, typename VectorType>
class DealIIMeshEvaluatorDevice
{
public:
  using mesh_type = DealIIMesh<dim>;
  using vector_type = VectorType;
  using value_type = typename VectorType::value_type;
  using operator_type = Operator<vector_type>;
  using global_operator_type = SparseMatrixDeviceOperator<vector_type>;
  using local_operator_type = SparseMatrixDeviceOperator<vector_type>;

  DealIIMeshEvaluatorDevice(cusolverDnHandle_t cusolver_dn_handle_,
                            cusolverSpHandle_t cusolver_sp_handle_,
                            cusparseHandle_t cusparse_handle_)
      : cusolver_dn_handle(cusolver_dn_handle_),
        cusolver_sp_handle(cusolver_sp_handle_),
        cusparse_handle(cusparse_handle_)
  {
  }

  std::shared_ptr<global_operator_type const>
  get_global_operator(mesh_type &mesh) const;

  std::shared_ptr<global_operator_type const>
  get_local_operator(mesh_type &mesh) const;

  virtual dealii::LinearAlgebra::distributed::Vector<value_type>
  get_locally_relevant_diag() const = 0;

  cusolverDnHandle_t cusolver_dn_handle;
  cusolverSpHandle_t cusolver_sp_handle;
  cusparseHandle_t cusparse_handle;

protected:
  virtual void
  evaluate_local(dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
                 std::shared_ptr<SparseMatrixDevice<value_type>> &) const = 0;

  virtual void
  evaluate_global(dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
                  std::shared_ptr<SparseMatrixDevice<value_type>> &) const = 0;
};

template <int dim, typename VectorType>
std::shared_ptr<typename DealIIMeshEvaluatorDevice<
    dim, VectorType>::global_operator_type const>
DealIIMeshEvaluatorDevice<dim, VectorType>::get_global_operator(
    mesh_type &mesh) const
{
  using matrix_type = typename global_operator_type::matrix_type;

  auto system_matrix = std::make_shared<matrix_type>();

  evaluate_global(mesh._dof_handler, mesh._constraints, system_matrix);

  return std::make_shared<global_operator_type>(system_matrix);
}

template <int dim, typename VectorType>
std::shared_ptr<typename DealIIMeshEvaluatorDevice<
    dim, VectorType>::global_operator_type const>
DealIIMeshEvaluatorDevice<dim, VectorType>::get_local_operator(
    mesh_type &mesh) const
{
  using matrix_type = typename global_operator_type::matrix_type;

  auto system_matrix = std::make_shared<matrix_type>();

  evaluate_local(mesh._dof_handler, mesh._constraints, system_matrix);

  return std::make_shared<global_operator_type>(system_matrix);
}

//---------------------------------------------------------------------------//

template <int dim, typename VectorType>
class Adapter<DealIIMeshEvaluatorDevice<dim, VectorType>>
{
public:
  using mesh_evaluator_type = DealIIMeshEvaluatorDevice<dim, VectorType>;
  using mesh_type = typename mesh_evaluator_type::mesh_type;
  using operator_type = typename mesh_evaluator_type::operator_type;
  using global_operator_type =
      typename mesh_evaluator_type::global_operator_type;
  using smoother_type = SmootherDeviceOperator<VectorType>;
  using direct_solver_type = DirectDeviceOperator<VectorType>;
  using vector_type = VectorType;

  static std::shared_ptr<operator_type>
  build_restrictor(MPI_Comm comm, mesh_evaluator_type const &evaluator,
                   mesh_type &mesh,
                   std::shared_ptr<boost::property_tree::ptree> params)
  {
    auto eigensolver_params = params->get_child("eigensolver");
    AMGe_device<mesh_type::dimension(), mesh_evaluator_type, vector_type> amge(
        comm, mesh._dof_handler, evaluator.cusolver_dn_handle,
        evaluator.cusparse_handle);

    std::array<unsigned int, dim> agglomerate_dim;
    auto agglomerate_params = params->get_child("agglomeration");
    agglomerate_dim[0] = agglomerate_params.get<unsigned int>("nx");
    agglomerate_dim[1] = agglomerate_params.get<unsigned int>("ny");
    if (dim == 3)
      agglomerate_dim[2] = agglomerate_params.get<unsigned int>("nz");
    int n_eigenvectors = eigensolver_params.get("number of eigenvectors", 1);
    double tolerance = eigensolver_params.get("tolerance", 1e-14);

    auto global_operator = evaluator.get_global_operator(mesh);

    auto restrictor_matrix =
        std::make_shared<typename global_operator_type::matrix_type>(
            amge.setup_restrictor(agglomerate_dim, n_eigenvectors, tolerance,
                                  evaluator, global_operator));

    return std::make_shared<global_operator_type>(restrictor_matrix);
  }

  static std::shared_ptr<operator_type>
  build_smoother(operator_type const &op,
                 std::shared_ptr<boost::property_tree::ptree> params)
  {
    auto global_op = dynamic_cast<global_operator_type const &>(op);
    return std::make_shared<smoother_type>(*global_op.get_matrix(), params);
  }

  static std::shared_ptr<operator_type>
  build_direct_solver(operator_type const &op,
                      mesh_evaluator_type const &evaluator,
                      std::shared_ptr<boost::property_tree::ptree> params)
  {
    auto global_op = dynamic_cast<global_operator_type const &>(op);

    return std::make_shared<direct_solver_type>(
        evaluator.cusolver_dn_handle, evaluator.cusolver_sp_handle,
        *global_op.get_matrix(), params);
  }
};
}

#endif
