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

#ifndef MFMG_LAPLACE_MATRIX_FREE_DEVICE_CUH
#define MFMG_LAPLACE_MATRIX_FREE_DEVICE_CUH

#include <mfmg/cuda/utils.cuh>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/affine_constraints.templates.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/vector_memory.templates.h>
#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/property_tree/ptree.hpp>

#include "laplace_matrix_free.hpp"

template <int dim, int fe_degree, class MaterialProperty>
class CoefficientFunctor
{
public:
  CoefficientFunctor(MaterialProperty const &material_property,
                     double *coefficient)
      : _material_property(material_property), _coef(coefficient)
  {
  }
  __device__ void
  operator()(const unsigned int cell,
             const typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data
                 *gpu_data);
  static const unsigned int n_dofs_1d = fe_degree + 1;
  static const unsigned int n_local_dofs =
      dealii::Utilities::pow(n_dofs_1d, dim);
  static const unsigned int n_q_points = dealii::Utilities::pow(n_dofs_1d, dim);

private:
  MaterialProperty const &_material_property;
  double *_coef;
};
template <int dim, int fe_degree, class MaterialProperty>
__device__ void CoefficientFunctor<dim, fe_degree, MaterialProperty>::
operator()(const unsigned int cell,
           const typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data
               *gpu_data)
{
  unsigned int const pos = dealii::CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);
  auto const point = dealii::CUDAWrappers::get_quadrature_point<dim, double>(
      cell, gpu_data, n_dofs_1d);
  _coef[pos] = _material_property.value(point);
}

template <int dim, int fe_degree, typename ScalarType>
class LaplaceOperatorQuad
{
public:
  __device__ LaplaceOperatorQuad(ScalarType coef) : _coef(coef) {}

  __device__ void
  operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1,
                                                1, ScalarType> *fe_eval,
             const unsigned int q) const;

private:
  ScalarType _coef;
};

template <int dim, int fe_degree, typename ScalarType>
__device__ void LaplaceOperatorQuad<dim, fe_degree, ScalarType>::
operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1,
                                              ScalarType> *fe_eval,
           const unsigned int q) const
{
  dealii::Tensor<1, dim, ScalarType> tmp = fe_eval->get_gradient(q);
  tmp *= _coef;
  fe_eval->submit_gradient(tmp, q);
}

template <int dim, int fe_degree, typename ScalarType>
class LocalLaplaceOperator
{
public:
  LocalLaplaceOperator(ScalarType *coefficient) : _coef(coefficient) {}

  __device__ void operator()(
      unsigned int const cell,
      typename dealii::CUDAWrappers::MatrixFree<dim, ScalarType>::Data const
          *gpu_data,
      dealii::CUDAWrappers::SharedData<dim, ScalarType> *shared_data,
      ScalarType const *src, ScalarType *dst) const;

  static unsigned int const n_dofs_1d = fe_degree + 1;
  static unsigned int const n_local_dofs =
      dealii::Utilities::pow(n_dofs_1d, dim);
  static unsigned int const n_q_points = n_local_dofs;

private:
  ScalarType *_coef;
};

template <int dim, int fe_degree, typename ScalarType>
class LocalDiagonalLaplaceOperator
{
public:
  LocalDiagonalLaplaceOperator(ScalarType *coefficient) : _coef(coefficient) {}

  __device__ void operator()(
      unsigned int const cell,
      typename dealii::CUDAWrappers::MatrixFree<dim, ScalarType>::Data const
          *gpu_data,
      dealii::CUDAWrappers::SharedData<dim, ScalarType> *shared_data,
      ScalarType const *src, ScalarType *dst) const;

  static unsigned int const n_dofs_1d = fe_degree + 1;
  static unsigned int const n_local_dofs =
      dealii::Utilities::pow(n_dofs_1d, dim);
  static unsigned int const n_q_points = n_local_dofs;

private:
  ScalarType *_coef;
};

template <int dim, int fe_degree, typename ScalarType>
__device__ void LocalLaplaceOperator<dim, fe_degree, ScalarType>::operator()(
    const unsigned int cell,
    const typename dealii::CUDAWrappers::MatrixFree<dim, ScalarType>::Data
        *gpu_data,
    dealii::CUDAWrappers::SharedData<dim, ScalarType> *shared_data,
    ScalarType const *src, ScalarType *dst) const
{
  const unsigned int pos = dealii::CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);

  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1,
                                     ScalarType>
      fe_eval(cell, gpu_data, shared_data);
  fe_eval.read_dof_values(src);
  bool const evaluate_values = false;
  bool const evaluate_gradients = true;
  bool const integrate_values = false;
  bool const integrate_gradients = true;
  fe_eval.evaluate(evaluate_values, evaluate_gradients);
  auto value = _coef[pos];
  LaplaceOperatorQuad<dim, fe_degree, ScalarType> dummy(value);
  fe_eval.apply_quad_point_operations(dummy);
  fe_eval.integrate(integrate_values, integrate_gradients);
  fe_eval.distribute_local_to_global(dst);
}

template <int dim, int fe_degree, typename ScalarType>
__device__ void LocalDiagonalLaplaceOperator<dim, fe_degree, ScalarType>::
operator()(
    const unsigned int cell,
    const typename dealii::CUDAWrappers::MatrixFree<dim, ScalarType>::Data
        *gpu_data,
    dealii::CUDAWrappers::SharedData<dim, ScalarType> *shared_data,
    ScalarType const * /*src*/, ScalarType *dst) const
{
  const unsigned int pos = dealii::CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);

  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1,
                                     ScalarType>
      fe_eval(cell, gpu_data, shared_data);

  ScalarType my_diagonal = 0.0;
  const unsigned int n_q_points_1d = n_dofs_1d;

  const unsigned int tid =
      (threadIdx.x % n_q_points_1d) +
      (dim > 1 ? threadIdx.y : 0) * n_q_points_1d +
      (dim > 2 ? threadIdx.z : 0) * n_q_points_1d * n_q_points_1d;

  bool const evaluate_values = false;
  bool const evaluate_gradients = true;
  bool const integrate_values = false;
  bool const integrate_gradients = true;

  for (unsigned int i = 0; i < n_local_dofs; ++i)
  {
    // set to unit vector
    fe_eval.submit_dof_value(i == tid ? 1.0 : 0.0, tid);
    __syncthreads();
    fe_eval.evaluate(evaluate_values, evaluate_gradients);
    auto value = _coef[pos];
    fe_eval.apply_quad_point_operations(
        LaplaceOperatorQuad<dim, fe_degree, ScalarType>{value});
    fe_eval.integrate(integrate_values, integrate_gradients);
    if (tid == i)
      my_diagonal = fe_eval.get_dof_value(tid);
  }
  __syncthreads();
  fe_eval.submit_dof_value(my_diagonal, tid);
  fe_eval.distribute_local_to_global(dst);
}

template <int dim, int fe_degree, typename ScalarType>
class LaplaceOperatorDevice
{
public:
  typedef ScalarType value_type;

  LaplaceOperatorDevice(
      MPI_Comm const &comm, dealii::DoFHandler<dim> const &dof_handler,
      dealii::AffineConstraints<ScalarType> const &constraints);

  void evaluate_coefficient(Coefficient<dim> const &material_property);

  void vmult(dealii::LinearAlgebra::distributed::Vector<
                 ScalarType, dealii::MemorySpace::CUDA> &dst,
             const dealii::LinearAlgebra::distributed::Vector<
                 ScalarType, dealii::MemorySpace::CUDA> &src) const;

  std::shared_ptr<
      dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<
          ScalarType, dealii::MemorySpace::CUDA>>>
  get_matrix_diagonal() const
  {
    return _diagonal;
  }

  std::shared_ptr<
      dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<
          ScalarType, dealii::MemorySpace::CUDA>>>
  get_matrix_diagonal_inverse() const
  {
    return _diagonal_inverse;
  }

  void compute_diagonal(const dealii::IndexSet &locally_relevant_dofs,
                        const dealii::IndexSet &locally_owned_dofs,
const MPI_Comm &mpi_communicator);

private:
  dealii::CUDAWrappers::MatrixFree<dim, ScalarType> _mf_data;

  std::shared_ptr<
      dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<
          ScalarType, dealii::MemorySpace::CUDA>>>
      _diagonal;

  std::shared_ptr<
      dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<
          double, dealii::MemorySpace::CUDA>>>
      _diagonal_inverse;

  dealii::LinearAlgebra::CUDAWrappers::Vector<double> _coef;
};

template <int dim, int fe_degree, typename ScalarType>
LaplaceOperatorDevice<dim, fe_degree, ScalarType>::LaplaceOperatorDevice(
    MPI_Comm const &comm, dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<ScalarType> const &constraints)
{
  dealii::MappingQGeneric<dim> mapping(fe_degree);
  typename dealii::CUDAWrappers::MatrixFree<dim, ScalarType>::AdditionalData
      additional_data;
  additional_data.mapping_update_flags = dealii::update_gradients |
                                         dealii::update_JxW_values |
                                         dealii::update_quadrature_points;
  const dealii::QGauss<1> quad(fe_degree + 1);
  _mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);

  auto *parallel_triangulation_ptr =
      dynamic_cast<const dealii::parallel::Triangulation<dim> *>(
          &dof_handler.get_triangulation());

  const unsigned int n_owned_cells =
      parallel_triangulation_ptr
          ? parallel_triangulation_ptr->n_locally_owned_active_cells()
          : dof_handler.get_triangulation().n_active_cells();

  _coef.reinit(dealii::Utilities::pow(fe_degree + 1, dim) * n_owned_cells);
}

template <int dim, int fe_degree, typename ScalarType>
void LaplaceOperatorDevice<dim, fe_degree, ScalarType>::evaluate_coefficient(
    Coefficient<dim> const &material_property)
{
  const CoefficientFunctor<dim, fe_degree, Coefficient<dim>> functor(
      material_property, _coef.get_values());
  _mf_data.evaluate_coefficients(functor);
}

template <int dim, int fe_degree, typename ScalarType>
void LaplaceOperatorDevice<dim, fe_degree, ScalarType>::vmult(
    dealii::LinearAlgebra::distributed::Vector<ScalarType,
                                               dealii::MemorySpace::CUDA> &dst,
    const dealii::LinearAlgebra::distributed::Vector<
        ScalarType, dealii::MemorySpace::CUDA> &src) const
{
  dst = 0.;
  LocalLaplaceOperator<dim, fe_degree, ScalarType> local_laplace_operator(
      _coef.get_values());
  _mf_data.cell_loop(local_laplace_operator, src, dst);
  _mf_data.copy_constrained_values(src, dst);
}

template <typename Number>
__global__ void initialize_with_inverted_entries(Number *dst, Number const *src,
                                                 unsigned int size)
{
  const unsigned int dof =
      threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  if (dof < size)
    dst[dof] = 1. / src[dof];
}

template <int dim, int fe_degree, typename ScalarType>
void LaplaceOperatorDevice<dim, fe_degree, ScalarType>::compute_diagonal(
    const dealii::IndexSet &locally_owned_dofs,
    const dealii::IndexSet &locally_relevant_dofs,
    const MPI_Comm & mpi_communicator)
{
  _diagonal.reset(
      new dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<
          ScalarType, dealii::MemorySpace::CUDA>>());
  dealii::LinearAlgebra::distributed::Vector<
      ScalarType, dealii::MemorySpace::CUDA> &diagonal =
      _diagonal->get_vector();
 diagonal.reinit(locally_owned_dofs, locally_relevant_dofs,
                          mpi_communicator);

  // TODO _mf_data.initialize_dof_vector(diagonal);
  dealii::LinearAlgebra::distributed::Vector<ScalarType,
                                             dealii::MemorySpace::CUDA>
      dummy;
  LocalDiagonalLaplaceOperator<dim, fe_degree, ScalarType>
      local_diagonal_laplace_operator(_coef.get_values());
  _mf_data.cell_loop(local_diagonal_laplace_operator, dummy, diagonal);

  _mf_data.set_constrained_values(1., diagonal);

  _diagonal_inverse.reset(
      new dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<
          ScalarType, dealii::MemorySpace::CUDA>>());
  dealii::LinearAlgebra::distributed::Vector<
      ScalarType, dealii::MemorySpace::CUDA> &inverse_diagonal =
      _diagonal_inverse->get_vector();
 inverse_diagonal.reinit(locally_owned_dofs, locally_relevant_dofs,
                         mpi_communicator);
  // TODO _mf_data.initialize_dof_vector(inverse_diagonal);
  int const n_blocks = 1 + (diagonal.local_size() - 1) / mfmg::block_size;
  initialize_with_inverted_entries<<<n_blocks, mfmg::block_size>>>(
      inverse_diagonal.get_values(), diagonal.get_values(),
      diagonal.local_size());
}

template <int dim, int fe_degree, typename ScalarType>
class LaplaceMatrixFreeDevice
{
public:
  LaplaceMatrixFreeDevice(MPI_Comm const &comm);

  template <typename MaterialPropertyType>
  void setup_system(boost::property_tree::ptree const &ptree,
                    MaterialPropertyType const &material_property);

  template <typename SourceType>
  void assemble_rhs(SourceType const &source);

  template <typename PreconditionerType>
  void solve(PreconditionerType &preconditioner);

  double compute_error(dealii::Function<dim> const &exact_solution);

  // The following variable should be private but there are public for
  // simplicity
  MPI_Comm _comm;
  dealii::parallel::distributed::Triangulation<dim> _triangulation;
  dealii::FE_Q<dim> _fe;
  dealii::DoFHandler<dim> _dof_handler;
  dealii::IndexSet _locally_owned_dofs;
  dealii::IndexSet _locally_relevant_dofs;
  dealii::AffineConstraints<double> _constraints;
  std::unique_ptr<LaplaceOperatorDevice<dim, fe_degree, ScalarType>>
      _laplace_operator;
  dealii::LinearAlgebra::distributed::Vector<ScalarType,
                                             dealii::MemorySpace::CUDA>
      _solution;
  dealii::LinearAlgebra::distributed::Vector<ScalarType,
                                             dealii::MemorySpace::CUDA>
      _system_rhs;
};

template <int dim, int fe_degree, typename ScalarType>
LaplaceMatrixFreeDevice<dim, fe_degree, ScalarType>::LaplaceMatrixFreeDevice(
    MPI_Comm const &comm)
    : _comm(comm), _triangulation(_comm), _fe(fe_degree),
      _dof_handler(_triangulation)
{
}

template <int dim, int fe_degree, typename ScalarType>
template <typename MaterialPropertyType>
void LaplaceMatrixFreeDevice<dim, fe_degree, ScalarType>::setup_system(
    boost::property_tree::ptree const &ptree,
    MaterialPropertyType const &material_property)
{
  std::string const mesh = ptree.get("mesh", "hyper_cube");
  if (mesh == "hyper_ball")
    dealii::GridGenerator::hyper_ball(_triangulation);
  else
    dealii::GridGenerator::hyper_cube(_triangulation);

  _triangulation.refine_global(ptree.get("n_refinements", 3));

  // Set the boundary id to one
  auto boundary_cells =
      dealii::filter_iterators(_triangulation.active_cell_iterators(),
                               dealii::IteratorFilters::LocallyOwnedCell(),
                               dealii::IteratorFilters::AtBoundary());
  for (auto &cell : boundary_cells)
  {
    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        cell->face(f)->set_boundary_id(1);
  }

  if (ptree.get("distort_random", false))
    dealii::GridTools::distort_random(0.2, _triangulation);

  _dof_handler.distribute_dofs(_fe);

  std::string const reordering = ptree.get("reordering", "None");
  if (reordering == "Reverse Cuthill-McKee")
    dealii::DoFRenumbering::Cuthill_McKee(_dof_handler, true);
  else if (reordering == "King")
    dealii::DoFRenumbering::boost::king_ordering(_dof_handler);
  else if (reordering == "Reverse minimum degree")
    dealii::DoFRenumbering::boost::minimum_degree(_dof_handler, true);
  else if (reordering == "Hierarchical")
    dealii::DoFRenumbering::hierarchical(_dof_handler);

  // Get the IndexSets
  _locally_owned_dofs = _dof_handler.locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler,
                                                  _locally_relevant_dofs);

  // Compute the constraints
  _constraints.clear();
  _constraints.reinit(_locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler, _constraints);
  dealii::VectorTools::interpolate_boundary_values(
      _dof_handler, 1, dealii::Functions::ZeroFunction<dim>(), _constraints);
  _constraints.close();

  _laplace_operator =
      std::make_unique<LaplaceOperatorDevice<dim, fe_degree, ScalarType>>(
          _comm, _dof_handler, _constraints);
  _laplace_operator->evaluate_coefficient(material_property);

  // Resize the vectors
  _solution.reinit(_locally_owned_dofs, _comm);
  // We need information about non-owned dofs for the call to
  // distribute_local_to_global. LaplaceOperatorDevice does not yet have a
  // function for initializing vectors suitably (opposed to LaplaceOperator).
  // Hence, just make all locally relevant dofs ghost entries.
  _system_rhs.reinit(_locally_owned_dofs, _locally_relevant_dofs, _comm);

  _laplace_operator->compute_diagonal(_locally_owned_dofs,
                                      _locally_relevant_dofs, _comm);
}

template <int dim, int fe_degree, typename ScalarType>
template <typename SourceType>
void LaplaceMatrixFreeDevice<dim, fe_degree, ScalarType>::assemble_rhs(
    SourceType const &source)
{
  // Build on the rhs on the host and then move the data to the device
  dealii::LinearAlgebra::distributed::Vector<ScalarType> system_rhs_host(
      _system_rhs.get_partitioner());
  system_rhs_host = 0;

  dealii::QGauss<dim> const quadrature(fe_degree + 1);
  dealii::FEValues<dim> fe_values(_fe, quadrature,
                                  dealii::update_values |
                                      dealii::update_quadrature_points |
                                      dealii::update_JxW_values);
  unsigned int const dofs_per_cell = _fe.dofs_per_cell;
  unsigned int const n_q_points = quadrature.size();
  dealii::Vector<double> cell_rhs(dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (auto cell :
       dealii::filter_iterators(_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    cell_rhs = 0;
    fe_values.reinit(cell);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      double const rhs_value =
          source.value(fe_values.quadrature_point(q_point));
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        cell_rhs(i) += rhs_value * fe_values.shape_value(i, q_point) *
                       fe_values.JxW(q_point);
      }
    }
    cell->get_dof_indices(local_dof_indices);
    _constraints.distribute_local_to_global(cell_rhs, local_dof_indices,
                                            system_rhs_host);
  }
  system_rhs_host.compress(dealii::VectorOperation::add);

  // Move the data to the device
  unsigned int const local_size = system_rhs_host.local_size();
  cudaError_t cuda_error =
      cudaMemcpy(_system_rhs.get_values(), system_rhs_host.get_values(),
                 local_size * sizeof(ScalarType), cudaMemcpyHostToDevice);
  mfmg::ASSERT_CUDA(cuda_error);
}

template <int dim, int fe_degree, typename ScalarType>
template <typename PreconditionerType>
void LaplaceMatrixFreeDevice<dim, fe_degree, ScalarType>::solve(
    PreconditionerType &preconditioner)
{
  dealii::SolverControl solver_control(_dof_handler.n_dofs(),
                                       1e-12 * _system_rhs.l2_norm());
  dealii::SolverCG<dealii::LinearAlgebra::distributed::Vector<
      double, dealii::MemorySpace::CUDA>>
      cg(solver_control);
  _solution = 0.;

  cg.solve(*_laplace_operator, _solution, _system_rhs, preconditioner);

  if (dealii::Utilities::MPI::this_mpi_process(_comm) == 0)
    std::cout << "Solved in " << solver_control.last_step() << " iterations."
              << std::endl;
}

template <int dim, int fe_degree, typename ScalarType>
double LaplaceMatrixFreeDevice<dim, fe_degree, ScalarType>::compute_error(
    dealii::Function<dim> const &exact_solution)
{
  dealii::QGauss<dim> const quadrature(fe_degree + 1);
  dealii::LinearAlgebra::distributed::Vector<ScalarType> solution_host(
      _locally_owned_dofs, _comm);
  // Move the data to the device
  unsigned int const local_size = solution_host.local_size();
  cudaError_t cuda_error =
      cudaMemcpy(solution_host.get_values(), _solution.get_values(),
                 local_size * sizeof(ScalarType), cudaMemcpyDeviceToHost);
  mfmg::ASSERT_CUDA(cuda_error);
  _constraints.distribute(solution_host);

  dealii::LinearAlgebra::distributed::Vector<ScalarType>
      locally_relevant_solution_host(_locally_owned_dofs,
                                     _locally_relevant_dofs, _comm);
  locally_relevant_solution_host = solution_host;

  dealii::Vector<double> difference(
      _dof_handler.get_triangulation().n_active_cells());
  dealii::VectorTools::integrate_difference(
      _dof_handler, locally_relevant_solution_host, exact_solution, difference,
      quadrature, dealii::VectorTools::L2_norm);

  return dealii::VectorTools::compute_global_error(
      _dof_handler.get_triangulation(), difference,
      dealii::VectorTools::L2_norm);
}

#endif
