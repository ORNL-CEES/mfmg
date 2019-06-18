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

#define BOOST_TEST_MODULE hierarchy_boost

#include <mfmg/common/hierarchy.hpp>
#include <mfmg/cuda/utils.cuh>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/read_write_vector.h>

#include <boost/property_tree/info_parser.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>

#include <random>

#include "laplace.hpp"
#include "laplace_matrix_free_device.cuh"
#include "main.cc"
#include "test_hierarchy_helpers_device.cuh"

namespace bdata = boost::unit_test::data;

template <int dim>
double test_mf(std::shared_ptr<boost::property_tree::ptree> params)
{
  using DVector =
      dealii::LinearAlgebra::distributed::Vector<double,
                                                 dealii::MemorySpace::CUDA>;
  using value_type = typename DVector::value_type;
  using MeshEvaluator = mfmg::CudaMatrixFreeMeshEvaluator<dim>;

  mfmg::CudaHandle cuda_handle;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  auto material_property =
      MaterialPropertyFactory<dim>::create_material_property(
          params->get<std::string>("material_property.type"));
  Source<dim> source;

  auto laplace_ptree = params->get_child("laplace");
  int constexpr fe_degree = 1;
  LaplaceMatrixFreeDevice<dim, fe_degree, value_type> mf_laplace(comm);
  mf_laplace.setup_system(laplace_ptree, *material_property);

  auto const locally_owned_dofs = mf_laplace._locally_owned_dofs;
  DVector solution(locally_owned_dofs, comm);
  DVector rhs(mf_laplace._system_rhs);

  dealii::LinearAlgebra::ReadWriteVector<value_type> rw_vector(
      locally_owned_dofs);
  std::default_random_engine generator;
  std::uniform_real_distribution<value_type> distribution(0., 1.);
  for (auto const index : locally_owned_dofs)
  {
    // Make the solution satisfy the Dirichlet conditions because these should
    // be treated outside the preconditioner
    if (mf_laplace._constraints.is_constrained(index))
      rw_vector[index] = 0.;
    else
      rw_vector[index] = distribution(generator);
  }
  solution.import(rw_vector, dealii::VectorOperation::insert);

  auto evaluator =
      std::make_shared<TestMFMeshEvaluator<dim, fe_degree, value_type>>(
          comm, mf_laplace._dof_handler, mf_laplace._constraints,
          *mf_laplace._laplace_operator, material_property, cuda_handle);
  mfmg::Hierarchy<DVector> hierarchy(comm, evaluator, params);

  auto const &laplace_operator = mf_laplace._laplace_operator;

  // We want to do 20 V-cycle iterations. The rhs of is zero.
  // Use D(istributed)Vector because deal has its own Vector class
  DVector residual(rhs);
  unsigned int const n_cycles = 20;
  std::vector<double> res(n_cycles + 1);

  laplace_operator->vmult(residual, solution);
  residual.sadd(-1., 1., rhs);
  auto const residual0_norm = residual.l2_norm();

  std::cout << std::scientific;
  pcout << "#0: " << 1.0 << std::endl;
  res[0] = 1.0;
  for (unsigned int i = 0; i < n_cycles; ++i)
  {
    hierarchy.apply(rhs, solution);

    laplace_operator->vmult(residual, solution);
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
  auto fe_degree = laplace_ptree.get<unsigned>("fe_degree", 1);
  Laplace<dim, DVector> laplace(comm, fe_degree);
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
      comm, laplace._dof_handler, laplace._constraints, fe_degree, a,
      material_property, cuda_handle));
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

BOOST_DATA_TEST_CASE(hierarchy_2d,
                     bdata::make<std::string>({"matrix_based", "matrix_free"}),
                     mesh_evaluator_type)
{
  unsigned int constexpr dim = 2;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  // We only supports Jacobi smoother on the device
  params->put("smoother.type", "Jacobi");

  if (mesh_evaluator_type == "matrix_based")
    test<dim>(params);
  else
    test_mf<dim>(params);
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
          double const tolerance_percent = 1e-6;
          if (mesh == std::string("hyper_cube"))
          {
            BOOST_CHECK_CLOSE(
                conv_rate,
                ref_solution[std::make_tuple(mesh, distort_random, reordering)],
                tolerance_percent);
          }
          else
            BOOST_CHECK_CLOSE(
                conv_rate,
                ref_solution[std::make_tuple(mesh, distort_random, reordering)],
                tolerance_percent);
        }
  }
}

#if MFMG_WITH_AMGX
BOOST_DATA_TEST_CASE(amgx,
                     bdata::make<std::string>({"matrix_based", "matrix_free"}),
                     mesh_evaluator_type)
{
  // We do not do as many tests as for the two-grid because AMGx will only
  // use multiple levels if the problem is large enough.
  unsigned int constexpr dim = 3;
  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  params->put("solver.type", "amgx");
  params->put("solver.config_file", "amgx_config_amg.json");

  params->put("eigensolver.type", "lapack");
  params->put("agglomeration.nz", 2);
  params->put("laplace.n_refinements", 5);
  // We only supports Jacobi smoother on the device
  params->put("smoother.type", "Jacobi");

  // Relative tolerance in %
  double const tolerance_percent = 5.;
  // The convergence rate for the two grid algorithm is 0.345914564 which is
  // much better than the multigrid.
  if (mesh_evaluator_type == "matrix_based")
  {
    double const ref_solution = 0.86418797066393482;
    double const conv_rate = test<dim>(params);
    BOOST_CHECK_CLOSE(conv_rate, ref_solution, tolerance_percent);
  }
  else
  {
    // The convergence is much better in the matrix-free case. We need to
    // compare the matrices produces but the main difference seems that AMGx
    // is not coarsening the matrix as aggressively in this case.
    double const ref_solution = 0.65059751966355139;
    double const conv_rate = test_mf<dim>(params);
    BOOST_CHECK_CLOSE(conv_rate, ref_solution, tolerance_percent);
  }
}
#endif
