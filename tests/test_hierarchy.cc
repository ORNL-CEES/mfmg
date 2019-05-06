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

#define BOOST_TEST_MODULE hierarchy

#include <mfmg/common/hierarchy.hpp>
#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <EpetraExt_MatrixMatrix.h>

#include <boost/property_tree/info_parser.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>

#include <random>

#include "laplace.hpp"
#include "laplace_matrix_free.hpp"
#include "main.cc"
#include "test_hierarchy_helpers.hpp"

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

template <typename MeshEvaluator>
double test_mf(std::shared_ptr<boost::property_tree::ptree> params)
{
  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;
  int constexpr dim = MeshEvaluator::_dim;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  auto material_property =
      MaterialPropertyFactory<dim>::create_material_property(
          params->get<std::string>("material_property.type"));
  Source<dim> source;

  auto laplace_ptree = params->get_child("laplace");
  int constexpr fe_degree = 1;
  LaplaceMatrixFree<dim, fe_degree, double> mf_laplace(comm);
  mf_laplace.setup_system(laplace_ptree, *material_property);

  auto const locally_owned_dofs = mf_laplace._locally_owned_dofs;
  DVector solution(locally_owned_dofs, comm);
  DVector rhs(mf_laplace._system_rhs);

  std::default_random_engine generator;
  std::uniform_real_distribution<typename DVector::value_type> distribution(0.,
                                                                            1.);
  for (auto const index : locally_owned_dofs)
  {
    // Make the solution satisfy the Dirichlet conditions because these should
    // be treated outside the preconditioner
    if (mf_laplace._constraints.is_constrained(index))
      solution[index] = 0.;
    else
      solution[index] = distribution(generator);
  }

  auto evaluator =
      std::make_shared<TestMFMeshEvaluator<dim, fe_degree, double>>(
          mf_laplace._dof_handler, mf_laplace._constraints,
          mf_laplace._laplace_operator, material_property);
  mfmg::Hierarchy<DVector> hierarchy(comm, evaluator, params);

  // We want to do 20 V-cycle iterations. The rhs of is zero.
  // Use D(istributed)Vector because deal has its own Vector class
  DVector residual(rhs);
  unsigned int const n_cycles = 20;
  std::vector<double> res(n_cycles + 1);

  mf_laplace._laplace_operator.vmult(residual, solution);
  residual.sadd(-1., 1., rhs);
  auto const residual0_norm = residual.l2_norm();

  std::cout << std::scientific;
  // pcout << "#0: " << 1.0 << std::endl;
  res[0] = 1.0;
  for (unsigned int i = 0; i < n_cycles; ++i)
  {
    hierarchy.apply(rhs, solution);

    mf_laplace._laplace_operator.vmult(residual, solution);
    residual.sadd(-1., 1., rhs);
    double rel_residual = residual.l2_norm() / residual0_norm;
    // pcout << "#" << i + 1 << ": " << rel_residual << std::endl;
    res[i + 1] = rel_residual;
  }

  double const conv_rate = res[n_cycles] / res[n_cycles - 1];
  pcout << "Convergence rate: " << std::fixed << std::setprecision(2)
        << conv_rate << std::endl;

  return conv_rate;
}

template <typename MeshEvaluator>
double test(std::shared_ptr<boost::property_tree::ptree> params)
{
  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;
  int constexpr dim = MeshEvaluator::_dim;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  auto material_property =
      MaterialPropertyFactory<dim>::create_material_property(
          params->get<std::string>("material_property.type"));
  Source<dim> source;

  auto laplace_ptree = params->get_child("laplace");
  auto fe_degree = laplace_ptree.get<unsigned>("fe_degree", 1);
  std::cout << "FE degree: " << fe_degree << std::endl;
  Laplace<dim, DVector> laplace(comm, fe_degree);
  laplace.setup_system(laplace_ptree);
  laplace.assemble_system(source, *material_property);

  auto const &a = laplace._system_matrix;
  auto const locally_owned_dofs = laplace._locally_owned_dofs;
  DVector solution(locally_owned_dofs, comm);
  DVector rhs(laplace._system_rhs);

  std::default_random_engine generator;
  std::uniform_real_distribution<typename DVector::value_type> distribution(0.,
                                                                            1.);
  for (auto const index : locally_owned_dofs)
  {
    // Make the solution satisfy the Dirichlet conditions because these should
    // be treated outside the preconditioner
    if (laplace._constraints.is_constrained(index))
      solution[index] = 0.;
    else
      solution[index] = distribution(generator);
  }

  auto evaluator = std::make_shared<TestMeshEvaluator<MeshEvaluator>>(
      laplace._dof_handler, laplace._constraints, fe_degree, a,
      material_property);
  mfmg::Hierarchy<DVector> hierarchy(comm, evaluator, params);

  // We want to do 20 V-cycle iterations. The rhs of is zero.
  // Use D(istributed)Vector because deal has its own Vector class
  DVector residual(rhs);
  unsigned int const n_cycles = 20;
  std::vector<double> res(n_cycles + 1);

  a.vmult(residual, solution);
  residual.sadd(-1., 1., rhs);
  auto const residual0_norm = residual.l2_norm();

  std::cout << std::scientific;
  // pcout << "#0: " << 1.0 << std::endl;
  res[0] = 1.0;
  for (unsigned int i = 0; i < n_cycles; ++i)
  {
    hierarchy.apply(rhs, solution);

    a.vmult(residual, solution);
    residual.sadd(-1., 1., rhs);
    double rel_residual = residual.l2_norm() / residual0_norm;
    // pcout << "#" << i + 1 << ": " << rel_residual << std::endl;
    res[i + 1] = rel_residual;
  }

  double const conv_rate = res[n_cycles] / res[n_cycles - 1];
  pcout << "Convergence rate: " << std::fixed << std::setprecision(2)
        << conv_rate << std::endl;

  return conv_rate;
}

BOOST_AUTO_TEST_CASE(benchmark)
{
  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);

  test<mfmg::DealIIMeshEvaluator<2>>(params);
}

BOOST_AUTO_TEST_CASE(benchmark_mf)
{
  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  params->put("smoother.type", "Chebyshev");

  test_mf<mfmg::DealIIMatrixFreeMeshEvaluator<2>>(params);
}

typedef std::tuple<mfmg::DealIIMeshEvaluator<2>,
                   mfmg::DealIIMatrixFreeMeshEvaluator<2>>
    mesh_evaluator_types;
BOOST_AUTO_TEST_CASE_TEMPLATE(ml, MeshEvaluator, mesh_evaluator_types)
{
  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  bool is_matrix_free = false;
  if (mfmg::is_matrix_free<MeshEvaluator>::value)
  {
    params->put("smoother.type", "Chebyshev");
    is_matrix_free = true;
  }

  double gold_rate = is_matrix_free ? test_mf<MeshEvaluator>(params)
                                    : test<MeshEvaluator>(params);

  params->put("coarse.type", "ml");
  params->put("coarse.params.smoother: type", "symmetric Gauss-Seidel");
  params->put("coarse.params.max levels", 1);
  params->put("coarse.params.coarse: type", "Amesos-KLU");

  double ml_rate = is_matrix_free ? test_mf<MeshEvaluator>(params)
                                  : test<MeshEvaluator>(params);

  BOOST_TEST(ml_rate == gold_rate, tt::tolerance(1e-9));

  params->put("coarse.params.max levels", 2);

  ml_rate = is_matrix_free ? test_mf<MeshEvaluator>(params)
                           : test<MeshEvaluator>(params);

  BOOST_TEST(ml_rate > gold_rate);
}

BOOST_DATA_TEST_CASE(
    hierarchy_3d,
    bdata::make({"hyper_cube", "hyper_ball"}) * bdata::make({false, true}) *
        bdata::make({"None", "Reverse Cuthill_McKee"}) *
        bdata::make<std::string>({"DealIIMeshEvaluator",
                                  "DealIIMatrixFreeMeshEvaluator"}) *
        bdata::make<std::string>({"arpack", "lanczos"}),
    mesh, distort_random, reordering, mesh_evaluator_type, eigensolver)
{
  // TODO investigate why there is large difference in convergence rate when
  // running in parallel.
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
  {
    int constexpr dim = 3;
    auto params = std::make_shared<boost::property_tree::ptree>();
    boost::property_tree::info_parser::read_info("hierarchy_input.info",
                                                 *params);

    bool const is_matrix_free =
        mesh_evaluator_type == "DealIIMatrixFreeMeshEvaluator";
    if (is_matrix_free)
    {
      params->put("smoother.type", "Chebyshev");
    }

    params->put("eigensolver.type", eigensolver);
    params->put("agglomeration.nz", 2);
    params->put("laplace.n_refinements", 2);
    params->put("laplace.mesh", mesh);
    params->put("laplace.distort_random", distort_random);
    params->put("laplace.reordering", reordering);

    if (is_matrix_free && eigensolver == "arpack")
    {
      // skipping since ARPACK eigensolver is not available in matrix-free mode
      std::cout << "skip\n";
      return;
    }

    double const conv_rate =
        is_matrix_free
            ? test_mf<mfmg::DealIIMatrixFreeMeshEvaluator<dim>>(params)
            : test<mfmg::DealIIMeshEvaluator<dim>>(params);

    std::string distort_random_str =
        (distort_random ? "distort" : "no_distort");
    std::string matrix_free_str =
        (is_matrix_free ? "matrix-free" : "matrix-full");

    // This is a gold standard test. Not the greatest but it makes sure we don't
    // break the code
    //   (geometry, distortion, reordering, eigensolver, matrix-free)
    std::map<std::tuple<std::string, std::string, std::string, std::string,
                        std::string>,
             double>
        ref_solution;
    // clang-format off
    ref_solution[std::make_tuple("hyper_cube" , "no_distort" , "None"                  , "arpack"  , "matrix-full")] = 0.0235237332;
    ref_solution[std::make_tuple("hyper_cube" , "no_distort" , "Reverse Cuthill_McKee" , "arpack"  , "matrix-full")] = 0.0235237332;
    ref_solution[std::make_tuple("hyper_cube" , "distort"    , "None"                  , "arpack"  , "matrix-full")] = 0.0220847464;
    ref_solution[std::make_tuple("hyper_cube" , "distort"    , "Reverse Cuthill_McKee" , "arpack"  , "matrix-full")] = 0.0220847464;
    ref_solution[std::make_tuple("hyper_ball" , "no_distort" , "None"                  , "arpack"  , "matrix-full")] = 0.1149021369;
    ref_solution[std::make_tuple("hyper_ball" , "no_distort" , "Reverse Cuthill_McKee" , "arpack"  , "matrix-full")] = 0.1149021369;
    ref_solution[std::make_tuple("hyper_ball" , "distort"    , "None"                  , "arpack"  , "matrix-full")] = 0.1023844058;
    ref_solution[std::make_tuple("hyper_ball" , "distort"    , "Reverse Cuthill_McKee" , "arpack"  , "matrix-full")] = 0.1023844058;
    ref_solution[std::make_tuple("hyper_cube" , "no_distort" , "None"                  , "lanczos" , "matrix-full")] = 0.0235237332;
    ref_solution[std::make_tuple("hyper_cube" , "no_distort" , "None"                  , "lanczos" , "matrix-free")] = 0.0880045475;
    ref_solution[std::make_tuple("hyper_cube" , "no_distort" , "Reverse Cuthill_McKee" , "lanczos" , "matrix-full")] = 0.0235237332;
    ref_solution[std::make_tuple("hyper_cube" , "no_distort" , "Reverse Cuthill_McKee" , "lanczos" , "matrix-free")] = 0.0880045475;
    ref_solution[std::make_tuple("hyper_cube" , "distort"    , "None"                  , "lanczos" , "matrix-full")] = 0.0220847463;
    ref_solution[std::make_tuple("hyper_cube" , "distort"    , "None"                  , "lanczos" , "matrix-free")] = 0.0859227356;
    ref_solution[std::make_tuple("hyper_cube" , "distort"    , "Reverse Cuthill_McKee" , "lanczos" , "matrix-full")] = 0.0220847463;
    ref_solution[std::make_tuple("hyper_cube" , "distort"    , "Reverse Cuthill_McKee" , "lanczos" , "matrix-free")] = 0.0859227356;
    ref_solution[std::make_tuple("hyper_ball" , "no_distort" , "None"                  , "lanczos" , "matrix-full")] = 0.1148148381;
    ref_solution[std::make_tuple("hyper_ball" , "no_distort" , "None"                  , "lanczos" , "matrix-free")] = 0.2981146185;
    ref_solution[std::make_tuple("hyper_ball" , "no_distort" , "Reverse Cuthill_McKee" , "lanczos" , "matrix-full")] = 0.1148148381;
    ref_solution[std::make_tuple("hyper_ball" , "no_distort" , "Reverse Cuthill_McKee" , "lanczos" , "matrix-free")] = 0.2981146185;
    ref_solution[std::make_tuple("hyper_ball" , "distort"    , "None"                  , "lanczos" , "matrix-full")] = 0.1024523994;
    ref_solution[std::make_tuple("hyper_ball" , "distort"    , "None"                  , "lanczos" , "matrix-free")] = 0.2990825376;
    ref_solution[std::make_tuple("hyper_ball" , "distort"    , "Reverse Cuthill_McKee" , "lanczos" , "matrix-full")] = 0.1024523994;
    ref_solution[std::make_tuple("hyper_ball" , "distort"    , "Reverse Cuthill_McKee" , "lanczos" , "matrix-free")] = 0.2990825376;
    // clang-format on

    BOOST_TEST(
        conv_rate ==
            ref_solution[std::make_tuple(mesh, distort_random_str, reordering,
                                         eigensolver, matrix_free_str)],
        tt::tolerance(1e-2));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(zoltan, MeshEvaluator, mesh_evaluator_types)
{
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
  {
    auto params = std::make_shared<boost::property_tree::ptree>();
    boost::property_tree::info_parser::read_info("hierarchy_input.info",
                                                 *params);

    bool constexpr is_matrix_free = mfmg::is_matrix_free<MeshEvaluator>::value;
    if (is_matrix_free)
    {
      params->put("smoother.type", "Chebyshev");
    }

    params->put("agglomeration.partitioner", "zoltan");
    params->put("agglomeration.n_agglomerates", 4);

    // This is a gold standard test. Not the greatest but it makes sure we don't
    // break the code
    double const ref_solution = is_matrix_free ? 0.895602663 : 0.836618927;
    double const conv_rate = is_matrix_free ? test_mf<MeshEvaluator>(params)
                                            : test<MeshEvaluator>(params);
    BOOST_TEST(conv_rate == ref_solution, tt::tolerance(1e-6));
  }
}

// n_local_rows passed to gimme_a_matrix() must be the same on all processes
dealii::TrilinosWrappers::SparseMatrix
gimme_a_matrix(unsigned int n_local_rows, unsigned int n_entries_per_row)
{
  unsigned int const comm_size =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  unsigned int const comm_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int const size = comm_size * n_local_rows;
  dealii::IndexSet parallel_partitioning(size);
  unsigned int const local_offset = comm_rank * n_local_rows;
  for (unsigned int i = 0; i < n_local_rows; ++i)
    parallel_partitioning.add_index(i + local_offset);
  parallel_partitioning.compress();
  dealii::TrilinosWrappers::SparseMatrix sparse_matrix(
      parallel_partitioning, parallel_partitioning, MPI_COMM_WORLD);

  unsigned int nnz = 0;
  std::default_random_engine generator;
  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    std::uniform_int_distribution<int> distribution(0, size - 1);
    std::set<int> column_indices;
    for (unsigned int j = 0; j < n_entries_per_row; ++j)
    {
      int column_index = distribution(generator);
      int row_index = i + local_offset;
      sparse_matrix.set(row_index, column_index,
                        static_cast<double>(row_index + column_index));
      column_indices.insert(column_index);
    }
    nnz += column_indices.size();
  }
  sparse_matrix.compress(dealii::VectorOperation::insert);
  return sparse_matrix;
}

dealii::TrilinosWrappers::SparseMatrix gimme_identity(unsigned int n_local_rows)
{
  unsigned int const comm_size =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  unsigned int const comm_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int const size = comm_size * n_local_rows;
  dealii::IndexSet parallel_partitioning(size);
  unsigned int const local_offset = comm_rank * n_local_rows;
  for (unsigned int i = 0; i < n_local_rows; ++i)
    parallel_partitioning.add_index(i + local_offset);
  parallel_partitioning.compress();

  dealii::TrilinosWrappers::SparseMatrix identity_matrix(
      parallel_partitioning, parallel_partitioning, MPI_COMM_WORLD);

  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    unsigned int const global_i = i + local_offset;
    identity_matrix.set(global_i, global_i, 1.);
  }
  identity_matrix.compress(dealii::VectorOperation::insert);
  return identity_matrix;
}

BOOST_AUTO_TEST_CASE(fast_multiply_transpose)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;
  int constexpr dim = mfmg::DealIIMeshEvaluator<2>::_dim;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  params->put("eigensolver.type", "lapack");
  auto material_property =
      MaterialPropertyFactory<dim>::create_material_property(
          params->get<std::string>("material_property.type"));
  Source<dim> source;
  auto laplace_ptree = params->get_child("laplace");

  // Compute the ref AP
  Laplace<dim, DVector> ref_laplace(comm, 1);
  ref_laplace.setup_system(laplace_ptree);
  ref_laplace.assemble_system(source, *material_property);

  auto ref_evaluator =
      std::make_shared<TestMeshEvaluator<mfmg::DealIIMeshEvaluator<2>>>(
          ref_laplace._dof_handler, ref_laplace._constraints, 1,
          ref_laplace._system_matrix, material_property);
  std::unique_ptr<mfmg::HierarchyHelpers<DVector>> ref_hierarchy_helpers(
      new mfmg::DealIIHierarchyHelpers<dim, DVector>());
  auto ref_a = ref_hierarchy_helpers->get_global_operator(ref_evaluator);
  auto ref_restrictor =
      ref_hierarchy_helpers->build_restrictor(comm, ref_evaluator, params);
  auto ref_ap = ref_a->multiply_transpose(ref_restrictor);
  auto ref_matrix =
      std::dynamic_pointer_cast<mfmg::DealIITrilinosMatrixOperator<DVector>>(
          ref_ap)
          ->get_matrix();

  // Compute the fast AP
  params->put("fast_ap", true);
  Laplace<dim, DVector> fast_laplace(comm, 1);
  fast_laplace.setup_system(laplace_ptree);
  fast_laplace.assemble_system(source, *material_property);

  auto fast_evaluator =
      std::make_shared<TestMeshEvaluator<mfmg::DealIIMeshEvaluator<2>>>(
          fast_laplace._dof_handler, fast_laplace._constraints, 1,
          fast_laplace._system_matrix, material_property);
  std::unique_ptr<mfmg::HierarchyHelpers<DVector>> fast_hierarchy_helpers(
      new mfmg::DealIIHierarchyHelpers<dim, DVector>());

  auto fast_restrictor =
      fast_hierarchy_helpers->build_restrictor(comm, fast_evaluator, params);

  auto fast_ap = fast_hierarchy_helpers->fast_multiply_transpose();
  auto fast_matrix =
      std::dynamic_pointer_cast<mfmg::DealIITrilinosMatrixOperator<DVector>>(
          fast_ap)
          ->get_matrix();

  // Compare the two matrices obtained
  for (unsigned int i = 0; i < ref_matrix->m(); ++i)
    for (unsigned int j = 0; j < ref_matrix->n(); ++j)
      if (ref_matrix->el(i, j) > 1e-10)
        BOOST_TEST(fast_matrix->el(i, j) == ref_matrix->el(i, j),
                   tt::tolerance(1e-9));
}

BOOST_AUTO_TEST_CASE(fast_multiply_transpose_mf)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;
  int constexpr dim = mfmg::DealIIMatrixFreeMeshEvaluator<2>::_dim;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  params->put("eigensolver.type", "lanczos");
  auto material_property =
      MaterialPropertyFactory<dim>::create_material_property(
          params->get<std::string>("material_property.type"));
  Source<dim> source;
  auto laplace_ptree = params->get_child("laplace");

  // Compute the ref AP
  Laplace<dim, DVector> ref_laplace(comm, 1);
  ref_laplace.setup_system(laplace_ptree);
  ref_laplace.assemble_system(source, *material_property);

  auto ref_evaluator =
      std::make_shared<TestMeshEvaluator<mfmg::DealIIMeshEvaluator<2>>>(
          ref_laplace._dof_handler, ref_laplace._constraints, 1,
          ref_laplace._system_matrix, material_property);
  std::unique_ptr<mfmg::HierarchyHelpers<DVector>> ref_hierarchy_helpers(
      new mfmg::DealIIHierarchyHelpers<dim, DVector>());
  auto ref_a = ref_hierarchy_helpers->get_global_operator(ref_evaluator);
  auto ref_restrictor =
      ref_hierarchy_helpers->build_restrictor(comm, ref_evaluator, params);
  auto ref_ap = ref_a->multiply_transpose(ref_restrictor);
  auto ref_matrix =
      std::dynamic_pointer_cast<mfmg::DealIITrilinosMatrixOperator<DVector>>(
          ref_ap)
          ->get_matrix();

  // Compute the fast AP
  params->put("fast_ap", true);
  int constexpr fe_degree = 1;
  LaplaceMatrixFree<dim, fe_degree, double> fast_laplace(comm);
  fast_laplace.setup_system(laplace_ptree, *material_property);

  auto fast_evaluator =
      std::make_shared<TestMFMeshEvaluator<2, fe_degree, double>>(
          fast_laplace._dof_handler, fast_laplace._constraints,
          fast_laplace._laplace_operator, material_property);
  std::unique_ptr<mfmg::HierarchyHelpers<DVector>> fast_hierarchy_helpers(
      new mfmg::DealIIMatrixFreeHierarchyHelpers<dim, DVector>());

  auto fast_restrictor =
      fast_hierarchy_helpers->build_restrictor(comm, fast_evaluator, params);

  auto fast_ap = fast_hierarchy_helpers->fast_multiply_transpose();
  auto fast_matrix =
      std::dynamic_pointer_cast<mfmg::DealIITrilinosMatrixOperator<DVector>>(
          fast_ap)
          ->get_matrix();

  // Compare the two matrices obtained
  for (unsigned int i = 0; i < ref_matrix->m(); ++i)
    for (unsigned int j = 0; j < ref_matrix->n(); ++j)
      if (ref_matrix->el(i, j) > 1e-10)
        BOOST_TEST(fast_matrix->el(i, j) == ref_matrix->el(i, j),
                   tt::tolerance(1e-9));
}
