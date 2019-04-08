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

#ifndef AMGE_HOST_TEMPLATES_HPP
#define AMGE_HOST_TEMPLATES_HPP

#include <mfmg/dealii/amge_host.hpp>
#include <mfmg/dealii/lanczos.templates.hpp>

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <EpetraExt_MatrixMatrix.h>

namespace mfmg
{

struct Identity
{
  template <typename VectorType>
  void vmult(VectorType &dst, const VectorType &src) const
  {
    dst = src;
  }
};

template <typename MeshEvaluator>
struct MatrixFreeAgglomerateOperator
{
  using size_type = typename MeshEvaluator::size_type;

  template <typename DoFHandler>
  MatrixFreeAgglomerateOperator(MeshEvaluator const &mesh_evaluator,
                                DoFHandler &dof_handler,
                                dealii::AffineConstraints<double> &constraints)
      : _mesh_evaluator(mesh_evaluator), _dof_handler(dof_handler),
        _constraints(constraints)
  {
  }

  void vmult(dealii::Vector<double> &dst,
             const dealii::Vector<double> &src) const
  {
    _mesh_evaluator.matrix_free_evaluate_agglomerate(_dof_handler, src, dst);
  }

  std::vector<double> get_diag_elements() const
  {
    return _mesh_evaluator.matrix_free_get_agglomerate_diagonal(_dof_handler,
                                                                _constraints);
  }

  size_type m() const { return _dof_handler.n_dofs(); }

  size_type n() const { return _dof_handler.n_dofs(); }

private:
  MeshEvaluator const &_mesh_evaluator;
  static int constexpr dim = MeshEvaluator::_dim;
  dealii::DoFHandler<dim> &_dof_handler;
  dealii::AffineConstraints<double> &_constraints;
};

template <int dim, typename MeshEvaluator, typename VectorType>
AMGe_host<dim, MeshEvaluator, VectorType>::AMGe_host(
    MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
    boost::property_tree::ptree const &eigensolver_params)
    : AMGe<dim, VectorType>(comm, dof_handler),
      _eigensolver_params(eigensolver_params)
{
}

namespace
{
template <typename AgglomerateOperator>
void lanczos_compute_eigenvalues_and_eigenvectors(
    unsigned int n_eigenvectors, double tolerance,
    boost::property_tree::ptree const &eigensolver_params,
    AgglomerateOperator const &agglomerate_operator,
    dealii::Vector<double> const &initial_guess,
    std::vector<std::complex<double>> &eigenvalues,
    std::vector<dealii::Vector<double>> &eigenvectors)
{
  boost::property_tree::ptree lanczos_params;
  lanczos_params.put("num_eigenpairs", n_eigenvectors);
  // We are having trouble with Lanczos when tolerance is too tight.
  // Typically, it results in spurious eigenvalues (so far, only noticed 0).
  // This seems to be the result of producing too many Lanczos vectors. For
  // hierarchy_3d tests, any tolerance below 1e-5 (e.g., 1e-6) produces this
  // problem. Thus, we try to work around it here. It is still unclear how
  // robust this is.
  lanczos_params.put("tolerance", std::max(tolerance, 1e-4));
  lanczos_params.put("max_iterations",
                     eigensolver_params.get("max_iterations", 200));
  lanczos_params.put("percent_overshoot",
                     eigensolver_params.get("percent_overshoot", 5));
  bool is_deflated = eigensolver_params.get("is_deflated", false);
  if (is_deflated)
  {
    lanczos_params.put("is_deflated", true);
    lanczos_params.put("num_cycles", eigensolver_params.get<int>("num_cycles"));
    lanczos_params.put("num_eigenpairs_per_cycle",
                       eigensolver_params.get<int>("num_eigenpairs_per_cycle"));
  }

  Lanczos<AgglomerateOperator, dealii::Vector<double>> solver(
      agglomerate_operator);

  std::vector<double> real_eigenvalues;
  std::tie(real_eigenvalues, eigenvectors) =
      solver.solve(lanczos_params, initial_guess);
  ASSERT(n_eigenvectors == eigenvectors.size(),
         "Wrong number of computed eigenpairs");

  // Copy real eigenvalues to complex
  std::copy(real_eigenvalues.begin(), real_eigenvalues.end(),
            eigenvalues.begin());
}
} // namespace

template <int dim, typename MeshEvaluator, typename VectorType>
template <typename Triangulation>
std::tuple<std::vector<std::complex<double>>,
           std::vector<dealii::Vector<double>>,
           std::vector<typename VectorType::value_type>,
           std::vector<dealii::types::global_dof_index>>
AMGe_host<dim, MeshEvaluator, VectorType>::compute_local_eigenvectors(
    unsigned int n_eigenvectors, double tolerance,
    Triangulation const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    MeshEvaluator const &evaluator,
    typename std::enable_if_t<is_matrix_free<MeshEvaluator>::value &&
                                  std::is_class<Triangulation>::value,
                              int>) const
{
  dealii::DoFHandler<dim> agglomerate_dof_handler(agglomerate_triangulation);
  dealii::AffineConstraints<double> agglomerate_constraints;

  using AgglomerateOperator = MatrixFreeAgglomerateOperator<MeshEvaluator>;
  AgglomerateOperator agglomerate_operator(evaluator, agglomerate_dof_handler,
                                           agglomerate_constraints);

  auto const diag_elements = agglomerate_operator.get_diag_elements();

  // Compute the eigenvalues and the eigenvectors
  unsigned int const n_dofs_agglomerate = agglomerate_operator.m();
  std::vector<std::complex<double>> eigenvalues(n_eigenvectors);
  std::vector<dealii::Vector<double>> eigenvectors(
      n_eigenvectors, dealii::Vector<double>(n_dofs_agglomerate));

  auto const eigensolver_type =
      _eigensolver_params.get<std::string>("type", "lanczos");
  dealii::Vector<double> initial_vector(n_dofs_agglomerate);
  evaluator.set_initial_guess(agglomerate_constraints, initial_vector);
  if (eigensolver_type == "lanczos")
  {
    lanczos_compute_eigenvalues_and_eigenvectors(
        n_eigenvectors, tolerance, _eigensolver_params, agglomerate_operator,
        initial_vector, eigenvalues, eigenvectors);
  }
  else if (eigensolver_type == "arpack")
  {
    throw std::runtime_error(
        "ARPACK not available as eigensolver in matrix-free mode");
  }
  else if (eigensolver_type == "lapack")
  {
    throw std::runtime_error(
        "LAPACK not available as eigensolver in matrix-free mode");
  }
  else
  {
    ASSERT(false, "Unknown eigensolver type '" + eigensolver_type + "'");
  }

  // Compute the map between the local and the global dof indices.
  std::vector<dealii::types::global_dof_index> dof_indices_map =
      this->compute_dof_index_map(patch_to_global_map, agglomerate_dof_handler);

  return std::make_tuple(eigenvalues, eigenvectors, diag_elements,
                         dof_indices_map);
}

template <int dim, typename MeshEvaluator, typename VectorType>
template <typename Triangulation>
std::tuple<std::vector<std::complex<double>>,
           std::vector<dealii::Vector<double>>,
           std::vector<typename VectorType::value_type>,
           std::vector<dealii::types::global_dof_index>>
AMGe_host<dim, MeshEvaluator, VectorType>::compute_local_eigenvectors(
    unsigned int n_eigenvectors, double tolerance,
    Triangulation const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    MeshEvaluator const &evaluator,
    typename std::enable_if_t<!is_matrix_free<MeshEvaluator>::value &&
                                  std::is_class<Triangulation>::value,
                              int>) const
{
  dealii::DoFHandler<dim> agglomerate_dof_handler(agglomerate_triangulation);
  dealii::AffineConstraints<double> agglomerate_constraints;
  using value_type = typename VectorType::value_type;
  dealii::SparsityPattern agglomerate_sparsity_pattern;
  dealii::SparseMatrix<value_type> agglomerate_system_matrix;

  // Call user function to build the system matrix
  evaluator.evaluate_agglomerate(
      agglomerate_dof_handler, agglomerate_constraints,
      agglomerate_sparsity_pattern, agglomerate_system_matrix);

  // Get the diagonal elements
  unsigned int const size = agglomerate_system_matrix.m();
  std::vector<ScalarType> diag_elements(size);
  for (unsigned int i = 0; i < size; ++i)
    diag_elements[i] = agglomerate_system_matrix.diag_element(i);

  // Shift eigenvalues away from zero
  const double average_diagonal =
      std::accumulate(diag_elements.begin(), diag_elements.end(), 0.) / size;
  for (unsigned int i = 0; i < size; ++i)
    agglomerate_system_matrix.diag_element(i) += average_diagonal;

  // Compute the eigenvalues and the eigenvectors
  unsigned int const n_dofs_agglomerate = agglomerate_system_matrix.m();
  std::vector<std::complex<double>> eigenvalues(n_eigenvectors);
  // Arpack only works with double not float
  std::vector<dealii::Vector<double>> eigenvectors(
      n_eigenvectors, dealii::Vector<double>(n_dofs_agglomerate));

  dealii::Vector<double> initial_vector(n_dofs_agglomerate);
  evaluator.set_initial_guess(agglomerate_constraints, initial_vector);
  auto const eigensolver_type =
      _eigensolver_params.get<std::string>("type", "arpack");
  if (eigensolver_type == "arpack")
  {
    // Make Identity mass matrix
    Identity agglomerate_mass_matrix;

    dealii::SparseDirectUMFPACK inv_system_matrix;
    inv_system_matrix.initialize(agglomerate_system_matrix);

    dealii::SolverControl solver_control(n_dofs_agglomerate, tolerance);
    unsigned int const n_arnoldi_vectors = 2 * n_eigenvectors + 2;
    bool const symmetric = true;
    // We want the eigenvalues of the smallest magnitudes but we need to ask
    // for the ones with the largest magnitudes because they are computed for
    // the inverse of the matrix we care about.
    auto const which_eigenvalues =
        dealii::ArpackSolver::WhichEigenvalues::largest_magnitude;
    dealii::ArpackSolver::AdditionalData additional_data(
        n_arnoldi_vectors, which_eigenvalues, symmetric);
    dealii::ArpackSolver solver(solver_control, additional_data);

    // Compute the eigenvectors. Arpack outputs eigenvectors with a L2 norm of
    // one.
    solver.set_initial_vector(initial_vector);
    solver.solve(agglomerate_system_matrix, agglomerate_mass_matrix,
                 inv_system_matrix, eigenvalues, eigenvectors);
  }
  else if (eigensolver_type == "lanczos")
  {
    lanczos_compute_eigenvalues_and_eigenvectors(
        n_eigenvectors, tolerance, _eigensolver_params,
        agglomerate_system_matrix, initial_vector, eigenvalues, eigenvectors);
  }
  else if (eigensolver_type == "lapack")
  {
    // Use Lapack to compute the eigenvalues
    dealii::LAPACKFullMatrix<double> full_matrix;
    full_matrix.copy_from(agglomerate_system_matrix);

    double const lower_bound = -0.5;
    double const upper_bound = 100.;
    double const tol = 1e-12;
    dealii::Vector<double> lapack_eigenvalues(size);
    dealii::FullMatrix<double> lapack_eigenvectors;
    full_matrix.compute_eigenvalues_symmetric(
        lower_bound, upper_bound, tol, lapack_eigenvalues, lapack_eigenvectors);

    // Copy the eigenvalues and the eigenvectors in the right format
    for (unsigned int i = 0; i < n_eigenvectors; ++i)
      eigenvalues[i] = lapack_eigenvalues[i];

    for (unsigned int i = 0; i < n_eigenvectors; ++i)
      for (unsigned int j = 0; j < n_dofs_agglomerate; ++j)
        eigenvectors[i][j] = lapack_eigenvectors[j][i];
  }
  else
  {
    ASSERT(false, "Unknown eigensolver type '" + eigensolver_type + "'");
  }

  // Shift eigenvalues back
  for (unsigned int i = 0; i < n_eigenvectors; ++i)
    eigenvalues[i] -= average_diagonal;

  // Compute the map between the local and the global dof indices.
  std::vector<dealii::types::global_dof_index> dof_indices_map =
      this->compute_dof_index_map(patch_to_global_map, agglomerate_dof_handler);

  return std::make_tuple(eigenvalues, eigenvectors, diag_elements,
                         dof_indices_map);
}

template <int dim, typename MeshEvaluator, typename VectorType>
void AMGe_host<dim, MeshEvaluator, VectorType>::setup_restrictor(
    boost::property_tree::ptree const &agglomerate_ptree,
    unsigned int const n_eigenvectors, double const tolerance,
    MeshEvaluator const &evaluator,
    dealii::LinearAlgebra::distributed::Vector<
        typename VectorType::value_type> const &locally_relevant_global_diag,
    dealii::TrilinosWrappers::SparseMatrix &restriction_sparse_matrix)
{
  // Flag the cells to build agglomerates.
  unsigned int const n_agglomerates =
      this->build_agglomerates(agglomerate_ptree);

  // Parallel part of the setup.
  std::vector<unsigned int> agglomerate_ids(n_agglomerates);
  std::iota(agglomerate_ids.begin(), agglomerate_ids.end(), 1);
  std::vector<dealii::Vector<double>> eigenvectors;
  std::vector<std::vector<ScalarType>> diag_elements;
  std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_maps;
  std::vector<unsigned int> n_local_eigenvectors;
  CopyData copy_data;
  dealii::WorkStream::run(
      agglomerate_ids.begin(), agglomerate_ids.end(),
      static_cast<
          std::function<void(std::vector<unsigned int>::iterator const &,
                             ScratchData &, CopyData &)>>(
          std::bind(&AMGe_host::local_worker, *this, n_eigenvectors, tolerance,
                    std::cref(evaluator), std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3)),
      static_cast<std::function<void(CopyData const &)>>(std::bind(
          &AMGe_host::copy_local_to_global, *this, std::placeholders::_1,
          std::ref(eigenvectors), std::ref(diag_elements),
          std::ref(dof_indices_maps), std::ref(n_local_eigenvectors))),
      ScratchData(), copy_data);

  AMGe<dim, VectorType>::compute_restriction_sparse_matrix(
      eigenvectors, diag_elements, dof_indices_maps, n_local_eigenvectors,
      locally_relevant_global_diag, restriction_sparse_matrix);
}

template <int dim, typename MeshEvaluator, typename VectorType>
void AMGe_host<dim, MeshEvaluator, VectorType>::local_worker(
    unsigned int const n_eigenvectors, double const tolerance,
    MeshEvaluator const &evaluator,
    std::vector<unsigned int>::iterator const &agg_id, ScratchData &,
    CopyData &copy_data)
{
  dealii::Triangulation<dim> agglomerate_triangulation;
  std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
           typename dealii::DoFHandler<dim>::active_cell_iterator>
      agglomerate_to_global_tria_map;

  this->build_agglomerate_triangulation(*agg_id, agglomerate_triangulation,
                                        agglomerate_to_global_tria_map);

  // We ignore the eigenvalues.
  std::tie(std::ignore, copy_data.local_eigenvectors, copy_data.diag_elements,
           copy_data.local_dof_indices_map) =
      compute_local_eigenvectors(n_eigenvectors, tolerance,
                                 agglomerate_triangulation,
                                 agglomerate_to_global_tria_map, evaluator);
}

template <int dim, typename MeshEvaluator, typename VectorType>
void AMGe_host<dim, MeshEvaluator, VectorType>::copy_local_to_global(
    CopyData const &copy_data,
    std::vector<dealii::Vector<double>> &eigenvectors,
    std::vector<std::vector<typename VectorType::value_type>> &diag_elements,
    std::vector<std::vector<dealii::types::global_dof_index>> &dof_indices_maps,
    std::vector<unsigned int> &n_local_eigenvectors)
{
  eigenvectors.insert(eigenvectors.end(), copy_data.local_eigenvectors.begin(),
                      copy_data.local_eigenvectors.end());

  diag_elements.push_back(copy_data.diag_elements);

  dof_indices_maps.push_back(copy_data.local_dof_indices_map);

  n_local_eigenvectors.push_back(copy_data.local_eigenvectors.size());
}
} // namespace mfmg

#endif
