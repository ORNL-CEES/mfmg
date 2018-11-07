/*************************************************************************
 * Copyright (c) 2017-2018 by the mfmg authors                           *
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
#include "main.cc"
#include "test_hierarchy_helpers.hpp"

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

template <int dim>
double test(std::shared_ptr<boost::property_tree::ptree> params)
{
  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;

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
  DVector rhs(laplace._system_rhs);

  std::default_random_engine generator;
  std::uniform_real_distribution<typename DVector::value_type> distribution(0.,
                                                                            1.);
  for (auto const index : locally_owned_dofs)
    solution[index] = distribution(generator);

  auto const mesh_evaluator_type =
      params->get<std::string>("mesh_evaluator_type", "DealIIMeshEvaluator");

  auto evaluator = std::make_shared<TestMeshEvaluator<dim>>(
      laplace._dof_handler, laplace._constraints, a, material_property,
      mesh_evaluator_type);
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

char const *mesh_evaluator_types[] = {"DealIIMeshEvaluator",
                                      "DealIIMatrixFreeMeshEvaluator"};
BOOST_DATA_TEST_CASE(benchmark, bdata::make(mesh_evaluator_types),
                     mesh_evaluator_type)
{
  unsigned int constexpr dim = 2;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  params->put("mesh_evaluator_type", mesh_evaluator_type);

  test<dim>(params);
}

BOOST_DATA_TEST_CASE(ml, bdata::make(mesh_evaluator_types), mesh_evaluator_type)
{
  unsigned int constexpr dim = 2;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  params->put("mesh_evaluator_type", mesh_evaluator_type);

  double gold_rate = test<dim>(params);

  params->put("coarse.type", "ml");
  params->put("coarse.params.smoother: type", "symmetric Gauss-Seidel");
  params->put("coarse.params.max levels", 1);
  params->put("coarse.params.coarse: type", "Amesos-KLU");

  double ml_rate = test<dim>(params);

  BOOST_TEST(ml_rate == gold_rate, tt::tolerance(1e-9));

  params->put("coarse.params.max levels", 2);

  ml_rate = test<dim>(params);

  BOOST_TEST(ml_rate > gold_rate);
}

BOOST_DATA_TEST_CASE(hierarchy_3d,
                     bdata::make({"hyper_cube", "hyper_ball"}) *
                         bdata::make({false, true}) *
                         bdata::make({"None", "Reverse Cuthill_McKee"}) *
                         bdata::make(mesh_evaluator_types),
                     mesh, distort_random, reordering, mesh_evaluator_type)
{
  // TODO investigate why there is large difference in convergence rate when
  // running in parallel.
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
  {
    unsigned int constexpr dim = 3;
    auto params = std::make_shared<boost::property_tree::ptree>();
    boost::property_tree::info_parser::read_info("hierarchy_input.info",
                                                 *params);
    params->put("mesh_evaluator_type", mesh_evaluator_type);

    params->put("eigensolver.type", "lapack");
    params->put("agglomeration.nz", 2);
    params->put("laplace.n_refinements", 2);
    params->put("laplace.mesh", mesh);
    params->put("laplace.distort_random", distort_random);
    params->put("laplace.reordering", reordering);

    double const conv_rate = test<dim>(params);

    // This is a gold standard test. Not the greatest but it makes sure we don't
    // break the code
    std::map<std::tuple<std::string, bool, std::string>, double> ref_solution;
    ref_solution[std::make_tuple("hyper_cube", false, "None")] = 0.0425111106;
    ref_solution[std::make_tuple("hyper_cube", false,
                                 "Reverse Cuthill_McKee")] = 0.0425111106;
    ref_solution[std::make_tuple("hyper_cube", true, "None")] = 0.0398672044;
    ref_solution[std::make_tuple("hyper_cube", true, "Reverse Cuthill_McKee")] =
        0.0398672044;
    ref_solution[std::make_tuple("hyper_ball", false, "None")] = 0.1303442282;
    ref_solution[std::make_tuple("hyper_ball", false,
                                 "Reverse Cuthill_McKee")] = 0.1303442282;
    ref_solution[std::make_tuple("hyper_ball", true, "None")] = 0.1431096468;
    ref_solution[std::make_tuple("hyper_ball", true, "Reverse Cuthill_McKee")] =
        0.1431096468;

    if (mesh == std::string("hyper_cube"))
      BOOST_TEST(
          conv_rate ==
              ref_solution[std::make_tuple(mesh, distort_random, reordering)],
          tt::tolerance(1e-6));
    else
      BOOST_TEST(
          conv_rate ==
              ref_solution[std::make_tuple(mesh, distort_random, reordering)],
          tt::tolerance(1e-6));
  }
}

// There is a problem with Zoltan where the second time it is called (with the
// same input) the partitioning is different than the first time. This looks
// like a problem from Zoltan (we only use it through deal.II and the code looks
// fine), so we run the Zoltan test only once.
BOOST_AUTO_TEST_CASE(zoltan)
{
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
  {
    unsigned int constexpr dim = 2;

    auto params = std::make_shared<boost::property_tree::ptree>();
    boost::property_tree::info_parser::read_info("hierarchy_input.info",
                                                 *params);
    params->put("mesh_evaluator_type", "DealIIMeshEvaluator");

    params->put("agglomeration.partitioner", "zoltan");
    params->put("agglomeration.n_agglomerates", 4);

    // This is a gold standard test. Not the greatest but it makes sure we don't
    // break the code
    double const ref_solution = 0.903284598;
    double const conv_rate = test<dim>(params);
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

BOOST_AUTO_TEST_CASE(matrix_transpose_matrix_multiply)
{
  //  unsigned int const n_local_rows = 10;
  //  unsigned int const n_entries_per_row = 3;
  //  auto A = gimme_a_matrix(n_local_rows, n_entries_per_row);
  //  auto B = gimme_a_matrix(n_local_rows, n_entries_per_row);
  //  dealii::TrilinosWrappers::SparseMatrix C(A.locally_owned_range_indices(),
  //                                           B.locally_owned_range_indices(),
  //                                           A.get_mpi_communicator());
  //  mfmg::matrix_transpose_matrix_multiply(C, B, A);
  //
  //  dealii::TrilinosWrappers::SparseMatrix
  //  BT(B.locally_owned_domain_indices(),
  //                                            B.locally_owned_range_indices(),
  //                                            B.get_mpi_communicator());
  //  // auto I = gimme_identity(n_local_rows);
  //  // B.Tmmult(BT, I);
  //  dealii::TrilinosWrappers::SparseMatrix
  //  C_ref(A.locally_owned_range_indices(),
  //                                               B.locally_owned_range_indices(),
  //                                               A.get_mpi_communicator());
  //  // A.mmult(C_ref, BT);
  //  int error_code = EpetraExt::MatrixMatrix::Multiply(
  //      A.trilinos_matrix(), false, B.trilinos_matrix(), true,
  //      const_cast<Epetra_CrsMatrix &>(BT.trilinos_matrix()));
  //  BOOST_TEST(error_code == 0);
  //  C_ref.reinit(BT.trilinos_matrix());
  //
  //  dealii::LinearAlgebra::distributed::Vector<double> u(
  //      A.locally_owned_domain_indices(), A.get_mpi_communicator());
  //  u = 1.;
  //
  //  dealii::LinearAlgebra::distributed::Vector<double> v(
  //      A.locally_owned_domain_indices(), A.get_mpi_communicator());
  //
  //  C.vmult(v, u);
  //  BOOST_TEST(v.l2_norm() > 1.);
  //
  //  u = -1.;
  //  C_ref.vmult_add(v, u);
  //  BOOST_TEST(v.l2_norm() == 0.);
}
