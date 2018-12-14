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

#define BOOST_TEST_MODULE hierarchy_boost

#include <mfmg/common/hierarchy.hpp>

#include <deal.II/base/conditional_ostream.h>

#include <boost/property_tree/info_parser.hpp>

#include <random>

#include "laplace.hpp"
#include "main.cc"

template <int dim>
class Source : public dealii::Function<dim>
{
public:
  Source() = default;

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override final
  {
    return 0.;
  }
};

template <int dim>
class ConstantMaterialProperty : public dealii::Function<dim>
{
public:
  ConstantMaterialProperty() = default;

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override final
  {
    return 1.;
  }
};

template <int dim>
class LinearXMaterialProperty : public dealii::Function<dim>
{
public:
  LinearXMaterialProperty() = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override final
  {
    return 1. + p[0];
  }
};

template <int dim>
class LinearMaterialProperty : public dealii::Function<dim>
{
public:
  LinearMaterialProperty() = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override final
  {
    double value = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      value += (1. + d) * p[d];

    return value;
  }
};

template <int dim>
class DiscontinuousMaterialProperty : public dealii::Function<dim>
{
public:
  DiscontinuousMaterialProperty() = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override final

  {
    double value = 10.;
    for (unsigned int d = 0; d < dim; ++d)
      if (p[d] > 0.5)
        value *= value;

    return value;
  }
};

template <int dim>
class MaterialPropertyFactory
{
public:
  static std::shared_ptr<dealii::Function<dim>>
  create_material_property(std::string const &material_type)
  {
    if (material_type == "constant")
      return std::make_shared<ConstantMaterialProperty<dim>>();
    else if (material_type == "linear_x")
      return std::make_shared<LinearXMaterialProperty<dim>>();
    else if (material_type == "linear")
      return std::make_shared<LinearMaterialProperty<dim>>();
    else if (material_type == "discontinuous")
      return std::make_shared<DiscontinuousMaterialProperty<dim>>();
    else
    {
      mfmg::ASSERT_THROW_NOT_IMPLEMENTED();

      return nullptr;
    }
  }
};

template <int dim>
class TestMeshEvaluator : public mfmg::CudaMeshEvaluator<dim>
{
public:
  TestMeshEvaluator(MPI_Comm comm, dealii::DoFHandler<dim> &dof_handler,
                    dealii::AffineConstraints<double> &constraints,
                    dealii::TrilinosWrappers::SparseMatrix const &matrix,
                    std::shared_ptr<dealii::Function<dim>> material_property,
                    mfmg::CudaHandle &cuda_handle)
      : mfmg::CudaMeshEvaluator<dim>(cuda_handle, dof_handler, constraints),
        _comm(comm), _matrix(matrix), _material_property(material_property)
  {
  }

  virtual dealii::LinearAlgebra::distributed::Vector<double>
  get_locally_relevant_diag() const override final
  {
    dealii::IndexSet locally_owned_dofs =
        _matrix.locally_owned_domain_indices();
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(this->_dof_handler,
                                                    locally_relevant_dofs);
    dealii::LinearAlgebra::distributed::Vector<double>
        locally_owned_global_diag(locally_owned_dofs, _comm);
    for (auto const val : locally_owned_dofs)
      locally_owned_global_diag[val] = _matrix.diag_element(val);
    locally_owned_global_diag.compress(dealii::VectorOperation::insert);

    dealii::LinearAlgebra::distributed::Vector<double>
        locally_relevant_global_diag(locally_owned_dofs, locally_relevant_dofs,
                                     _comm);
    locally_relevant_global_diag = locally_owned_global_diag;

    return locally_relevant_global_diag;
  }

  void evaluate_global(
      dealii::DoFHandler<dim> &, dealii::AffineConstraints<double> &,
      mfmg::SparseMatrixDevice<double> &system_matrix) const override final
  {
    system_matrix = std::move(mfmg::convert_matrix(_matrix));
    system_matrix.cusparse_handle = this->_cuda_handle.cusparse_handle;
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseCreateMatDescr(&system_matrix.descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(system_matrix.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(system_matrix.descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  }

  void evaluate_agglomerate(
      dealii::DoFHandler<dim> &dof_handler,
      dealii::AffineConstraints<double> &constraints,
      mfmg::SparseMatrixDevice<double> &system_matrix) const override final
  {
    unsigned int const fe_degree = 1;
    dealii::FE_Q<dim> fe(fe_degree);
    dof_handler.distribute_dofs(fe);

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);

    // Compute the constraints
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    dealii::VectorTools::interpolate_boundary_values(
        dof_handler, 1, dealii::Functions::ZeroFunction<dim>(), constraints);
    constraints.close();

    // Build the system sparsity pattern and reinitialize the system sparse
    // matrix
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    dealii::SparsityPattern agg_system_sparsity_pattern;
    agg_system_sparsity_pattern.copy_from(dsp);
    dealii::SparseMatrix<double> agg_system_matrix(agg_system_sparsity_pattern);

    // Fill the system matrix
    dealii::QGauss<dim> const quadrature(fe_degree + 1);
    dealii::FEValues<dim> fe_values(
        fe, quadrature,
        dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);
    unsigned int const dofs_per_cell = fe.dofs_per_cell;
    unsigned int const n_q_points = quadrature.size();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);
    for (auto cell :
         dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                  dealii::IteratorFilters::LocallyOwnedCell()))
    {
      cell_matrix = 0;
      fe_values.reinit(cell);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                 fe_values.shape_grad(j, q_point) *
                                 fe_values.JxW(q_point);

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, local_dof_indices,
                                             agg_system_matrix);
    }

    system_matrix = std::move(mfmg::convert_matrix(agg_system_matrix));
    system_matrix.cusparse_handle = this->_cuda_handle.cusparse_handle;
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseCreateMatDescr(&system_matrix.descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(system_matrix.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(system_matrix.descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  }

private:
  MPI_Comm _comm;
  dealii::TrilinosWrappers::SparseMatrix const &_matrix;
  std::shared_ptr<dealii::Function<dim>> _material_property;
};

template <int dim>
double test(std::shared_ptr<boost::property_tree::ptree> params)
{
  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;
  using MeshEvaluator = mfmg::CudaMeshEvaluator<dim>;

  mfmg::CudaHandle cuda_handle;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  auto material_property =
      MaterialPropertyFactory<dim>::create_material_property(
          params->get<std::string>("material_property.type"));
  Source<dim> source;

  auto laplace_ptree = params->get_child("laplace");
  Laplace<dim, DVector> laplace(comm, 1);
  laplace.setup_system(laplace_ptree);
  laplace.assemble_system(source, *material_property);

  auto const &a = laplace._system_matrix;
  auto const locally_owned_dofs = laplace._locally_owned_dofs;
  DVector solution(locally_owned_dofs, comm);
  DVector rhs(locally_owned_dofs, comm);
  rhs = laplace._system_rhs;

  std::default_random_engine generator;
  std::uniform_real_distribution<typename DVector::value_type> distribution(0.,
                                                                            1.);
  for (auto const index : locally_owned_dofs)
    solution[index] = distribution(generator);

  std::shared_ptr<MeshEvaluator> evaluator(new TestMeshEvaluator<dim>(
      comm, laplace._dof_handler, laplace._constraints, a, material_property,
      cuda_handle));
  mfmg::Hierarchy<DVector> hierarchy(comm, evaluator, params);

  pcout << "Grid complexity    : " << hierarchy.grid_complexity() << std::endl;
  pcout << "Operator complexity: " << hierarchy.operator_complexity()
        << std::endl;

  // We want to do 20 V-cycle iterations. The rhs of is zero.
  // Use D(istributed)Vector because deal has its own Vector class
  DVector residual(rhs);
  unsigned int const n_cycles = 20;
  std::vector<double> res(n_cycles + 1);

  a.vmult(residual, solution);
  residual.sadd(-1., 1., rhs);
  auto const residual0_norm = residual.l2_norm();

  std::cout << std::scientific;
  pcout << "#0: " << 1.0 << std::endl;
  res[0] = 1.0;
  for (unsigned int i = 0; i < n_cycles; ++i)
  {
    hierarchy.apply(rhs, solution);

    a.vmult(residual, solution);
    residual.sadd(-1., 1., rhs);
    double rel_residual = residual.l2_norm() / residual0_norm;
    pcout << "#" << i + 1 << ": " << rel_residual << std::endl;
    res[i + 1] = rel_residual;
  }

  double const conv_rate = res[n_cycles] / res[n_cycles - 1];
  pcout << "Convergence rate: " << std::fixed << std::setprecision(2)
        << conv_rate << std::endl;

  return conv_rate;
}

BOOST_AUTO_TEST_CASE(hierarchy_2d)
{
  unsigned int constexpr dim = 2;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  // We only supports Jacobi smoother on the device
  params->put("smoother.type", "Jacobi");

  test<dim>(params);
}

BOOST_AUTO_TEST_CASE(hierarchy_3d)
{
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
  {
    // This is gold standard test. Not the greatest but it makes sure we
    // don't break the code
    std::map<std::tuple<std::string, bool, std::string>, double> ref_solution;
    ref_solution[std::make_tuple("hyper_cube", false, "None")] =
        0.14933479171507894;
    ref_solution[std::make_tuple(
        "hyper_cube", false, "Reverse Cuthill_McKee")] = 0.14933479171507894;
    ref_solution[std::make_tuple("hyper_cube", true, "None")] =
        0.15334169107506268;
    ref_solution[std::make_tuple("hyper_cube", true, "Reverse Cuthill_McKee")] =
        0.15334169107506268;
    ref_solution[std::make_tuple("hyper_ball", false, "None")] =
        0.5953407021456707;
    ref_solution[std::make_tuple(
        "hyper_ball", false, "Reverse Cuthill_McKee")] = 0.59534070214567336;
    ref_solution[std::make_tuple("hyper_ball", true, "None")] =
        0.62020418011247469;
    ref_solution[std::make_tuple("hyper_ball", true, "Reverse Cuthill_McKee")] =
        0.62020418011247469;

    for (auto mesh : {"hyper_cube", "hyper_ball"})
      for (auto distort_random : {false, true})
        for (auto reordering : {"None", "Reverse Cuthill_McKee"})
        {
          unsigned int constexpr dim = 3;
          auto params = std::make_shared<boost::property_tree::ptree>();
          boost::property_tree::info_parser::read_info("hierarchy_input.info",
                                                       *params);
          params->put("solver.type", "lu_dense");

          params->put("eigensolver.type", "lapack");
          params->put("agglomeration.nz", 2);
          params->put("laplace.n_refinements", 2);
          params->put("laplace.mesh", mesh);
          params->put("laplace.distort_random", distort_random);
          params->put("laplace.reordering", reordering);
          // We only supports Jacobi smoother on the device
          params->put("smoother.type", "Jacobi");

          double const conv_rate = test<dim>(params);

          // Relative tolerance in %
          double const tolerance = 1e-6;
          if (mesh == std::string("hyper_cube"))
          {
            BOOST_CHECK_CLOSE(
                conv_rate,
                ref_solution[std::make_tuple(mesh, distort_random, reordering)],
                tolerance);
          }
          else
            BOOST_CHECK_CLOSE(
                conv_rate,
                ref_solution[std::make_tuple(mesh, distort_random, reordering)],
                tolerance);
        }
  }
}

#if MFMG_WITH_AMGX
BOOST_AUTO_TEST_CASE(amgx)
{
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
  {
    // We do not do as many tests as for the two-grid because AMGx will only
    // use multiple levels if the problem is large enough.
    unsigned int constexpr dim = 3;
    auto params = std::make_shared<boost::property_tree::ptree>();
    boost::property_tree::info_parser::read_info("hierarchy_input.info",
                                                 *params);
    params->put("solver.type", "amgx");
    params->put("solver.config_file", "amgx_config_amg.json");

    params->put("eigensolver.type", "lapack");
    params->put("agglomeration.nz", 2);
    params->put("laplace.n_refinements", 5);
    // We only supports Jacobi smoother on the device
    params->put("smoother.type", "Jacobi");

    double const tolerance = 1.;
    // The convergence rate for the two grid algorithm is 0.345914564 which is
    // much better than the multigrid.
    double const ref_solution = 0.86418797066393482;
    double const conv_rate = test<dim>(params);
    BOOST_CHECK_CLOSE(conv_rate, ref_solution, tolerance);
  }
}
#endif
