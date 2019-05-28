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

#include <mfmg/common/exceptions.hpp>

#include <deal.II/base/timer.h>

#include <boost/program_options.hpp>

#include "test_hierarchy_helpers.hpp"

template <int dim, int fe_degree>
void matrix_free_two_grids(std::shared_ptr<boost::property_tree::ptree> params)
{

  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  // Print a table with the timings when the destructor is called
  auto timer = std::make_shared<dealii::TimerOutput>(
      comm, pcout, dealii::TimerOutput::summary,
      dealii::TimerOutput::wall_times);

  auto material_property =
      MaterialPropertyFactory<dim>::create_material_property(
          params->get<std::string>("material_property.type"));
  Source<dim> source;

  auto const &laplace_ptree = params->get_child("laplace");
  LaplaceMatrixFree<dim, fe_degree, double> mf_laplace(comm);
  mf_laplace.setup_system(laplace_ptree, *material_property);

  auto const &locally_owned_dofs = mf_laplace._locally_owned_dofs;
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
  mfmg::Hierarchy<DVector> hierarchy(comm, evaluator, params, timer);

  // We want to do 20 V-cycle iterations. The rhs of is zero.
  // Use D(istributed)Vector because deal has its own Vector class
  DVector residual(rhs);
  unsigned int const n_cycles = 20;
  std::vector<double> res(n_cycles + 1);

  mf_laplace._laplace_operator.vmult(residual, solution);
  residual.sadd(-1., 1., rhs);
  auto const residual0_norm = residual.l2_norm();

  std::cout << std::scientific;
  res[0] = 1.0;
  for (unsigned int i = 0; i < n_cycles; ++i)
  {
    hierarchy.vmult(solution, rhs);

    mf_laplace._laplace_operator.vmult(residual, solution);
    residual.sadd(-1., 1., rhs);
    double rel_residual = residual.l2_norm() / residual0_norm;
    res[i + 1] = rel_residual;
  }

  double const conv_rate = res[n_cycles] / res[n_cycles - 1];
  pcout << "Convergence rate: " << std::fixed << std::setprecision(2)
        << conv_rate << std::endl;
}

template <int dim>
void matrix_based_two_grids(std::shared_ptr<boost::property_tree::ptree> params)
{
  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;
  using MeshEvaluator = mfmg::DealIIMeshEvaluator<dim>;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  // Print a table with the timings when the destructor is called
  auto timer = std::make_shared<dealii::TimerOutput>(
      comm, pcout, dealii::TimerOutput::summary,
      dealii::TimerOutput::wall_times);

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

  std::shared_ptr<MeshEvaluator> evaluator(
      new TestMeshEvaluator<mfmg::DealIIMeshEvaluator<dim>>(
          laplace._dof_handler, laplace._constraints, fe_degree, a,
          material_property));
  mfmg::Hierarchy<DVector> hierarchy(comm, evaluator, params, timer);

  // We want to do 20 V-cycle iterations. The rhs of is zero.
  // Use D(istributed)Vector because deal has its own Vector class
  DVector residual(rhs);
  unsigned int const n_cycles = 20;
  std::vector<double> res(n_cycles + 1);

  a.vmult(residual, solution);
  residual.sadd(-1., 1., rhs);
  auto const residual0_norm = residual.l2_norm();

  std::cout << std::scientific;
  res[0] = 1.0;
  for (unsigned int i = 0; i < n_cycles; ++i)
  {
    hierarchy.vmult(solution, rhs);

    a.vmult(residual, solution);
    residual.sadd(-1., 1., rhs);
    double rel_residual = residual.l2_norm() / residual0_norm;
    res[i + 1] = rel_residual;
  }

  double const conv_rate = res[n_cycles] / res[n_cycles - 1];
  pcout << "Convergence rate: " << std::fixed << std::setprecision(2)
        << conv_rate << std::endl;
}

int main(int argc, char *argv[])
{
  namespace boost_po = boost::program_options;

  MPI_Init(&argc, &argv);
  dealii::MultithreadInfo::set_thread_limit(1);

  boost_po::options_description cmd("Available options");
  cmd.add_options()("help,h", "produce help message");
  cmd.add_options()("filename,f", boost_po::value<std::string>()->multitoken(),
                    "file containing input parameters");
  cmd.add_options()("dim,d", boost_po::value<int>(), "dimension");
  cmd.add_options()("matrix_free,m", boost_po::value<bool>(),
                    "use matrix-free algorithm");

  boost_po::variables_map vm;
  boost_po::store(boost_po::parse_command_line(argc, argv, cmd), vm);
  boost_po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << cmd << std::endl;

    return 0;
  }

  std::string filename = {"hierarchy_input.info"};
  if (vm.count("filename"))
    filename = vm["filename"].as<std::string>();

  int dim = 2;
  if (vm.count("dim"))
    dim = vm["dim"].as<int>();
  mfmg::ASSERT(dim == 2 || dim == 3, "Dimension must be 2 or 3");

  bool matrix_free = false;
  if (vm.count("matrix_free"))
    matrix_free = vm["matrix_free"].as<bool>();

  auto const params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info(filename, *params);

  int const fe_degree = params->get<unsigned int>("laplace.fe_degree", 1);

  std::cout << "input file: " << filename << ", dimension: " << dim
            << ", matrix-free: " << matrix_free << ", fe_degree: " << fe_degree
            << std::endl;

  if (matrix_free)
  {
    if (dim == 2)
    {
      switch (fe_degree)
      {
      case 1:
      {
        matrix_free_two_grids<2, 1>(params);

        break;
      }
      case 2:
      {
        matrix_free_two_grids<2, 2>(params);

        break;
      }
      case 3:
      {
        matrix_free_two_grids<2, 3>(params);

        break;
      }
      case 4:
      {
        matrix_free_two_grids<2, 4>(params);

        break;
      }
      case 5:
      {
        matrix_free_two_grids<2, 5>(params);

        break;
      }
      case 6:
      {
        matrix_free_two_grids<2, 6>(params);

        break;
      }
      case 7:
      {
        matrix_free_two_grids<2, 7>(params);

        break;
      }
      case 8:
      {
        matrix_free_two_grids<2, 8>(params);

        break;
      }
      case 9:
      {
        matrix_free_two_grids<2, 9>(params);

        break;
      }
      case 10:
      {
        matrix_free_two_grids<2, 10>(params);

        break;
      }
      default:
      {
        mfmg::ASSERT(false, "The fe_degree should be between 1 and 10");
      }
      }
    }
    else
    {
      switch (fe_degree)
      {
      case 1:
      {
        matrix_free_two_grids<3, 1>(params);

        break;
      }
      case 2:
      {
        matrix_free_two_grids<3, 2>(params);

        break;
      }
      case 3:
      {
        matrix_free_two_grids<3, 3>(params);

        break;
      }
      case 4:
      {
        matrix_free_two_grids<3, 4>(params);

        break;
      }
      case 5:
      {
        matrix_free_two_grids<3, 5>(params);

        break;
      }
      case 6:
      {
        matrix_free_two_grids<3, 6>(params);

        break;
      }
      case 7:
      {
        matrix_free_two_grids<3, 7>(params);

        break;
      }
      case 8:
      {
        matrix_free_two_grids<3, 8>(params);

        break;
      }
      case 9:
      {
        matrix_free_two_grids<3, 9>(params);

        break;
      }
      case 10:
      {
        matrix_free_two_grids<3, 10>(params);

        break;
      }
      default:
      {
        mfmg::ASSERT(false, "The fe_degree should be between 1 and 10");
      }
      }
    }
  }
  else
  {
    if (dim == 2)
      matrix_based_two_grids<2>(params);
    else
      matrix_based_two_grids<3>(params);
  }

  MPI_Finalize();

  return 0;
}
