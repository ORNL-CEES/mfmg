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

#include "main.cc"

#include "laplace.hpp"

#include <mfmg/dealii_adapters.hpp>
#include <mfmg/hierarchy.hpp>

#include <test_hierarchy_helpers.hpp>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <boost/property_tree/info_parser.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>

#include <random>

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

template <int dim>
double test(std::shared_ptr<boost::property_tree::ptree> params)
{
  using DVector = dealii::TrilinosWrappers::MPI::Vector;
  using MeshEvaluator = mfmg::DealIIMeshEvaluator<dim, DVector>;
  using Mesh = mfmg::DealIIMesh<dim>;

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

  auto mesh =
      std::make_shared<Mesh>(laplace._dof_handler, laplace._constraints);

  auto const &a = laplace._system_matrix;
  auto const locally_owned_dofs = laplace._locally_owned_dofs;
  DVector solution(locally_owned_dofs, comm);
  DVector rhs(laplace._system_rhs);

  std::default_random_engine generator;
  std::uniform_real_distribution<typename DVector::value_type> distribution(0.,
                                                                            1.);
  for (auto const index : locally_owned_dofs)
    solution[index] = distribution(generator);

  TestMeshEvaluator<dim, DVector> evaluator(a, material_property);
  mfmg::Hierarchy<MeshEvaluator, DVector> hierarchy(comm, evaluator, *mesh,
                                                    params);

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

BOOST_AUTO_TEST_CASE(benchmark)
{
  unsigned int constexpr dim = 2;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);

  test<dim>(params);
}

BOOST_AUTO_TEST_CASE(ml)
{
  unsigned int constexpr dim = 2;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);

  double gold_rate = test<dim>(params);

  params->put("coarse.type", "ml");
  params->put("coarse.params.smoother: type", "symmetric Gauss-Seidel");
  params->put("coarse.params.max levels", 1);
  params->put("coarse.params.coarse: type", "Amesos-KLU");

  double ml_rate = test<dim>(params);

  BOOST_TEST(ml_rate == gold_rate, tt::tolerance(1e-9));

  params->put("coarse.params.max levels", 2);

  ml_rate = test<dim>(params);

  BOOST_TEST(ml_rate > gold_rate + 0.1);
}

BOOST_DATA_TEST_CASE(hierarchy_3d,
                     bdata::make({"hyper_cube", "hyper_ball"}) *
                         bdata::make({false, true}) *
                         bdata::make({"None", "Reverse Cuthill_McKee"}),
                     mesh, distort_random, reordering)
{
  // TODO investigate why there is large difference in convergence rate when
  // running in parallel.
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
  {
    unsigned int constexpr dim = 3;
    auto params = std::make_shared<boost::property_tree::ptree>();
    boost::property_tree::info_parser::read_info("hierarchy_input.info",
                                                 *params);

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
