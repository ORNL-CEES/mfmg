/*************************************************************************
 * Copyright (c) 2017 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#define BOOST_TEST_MODULE laplace

#include "laplace.hpp"
#include "main.cc"
#include <deal.II/lac/trilinos_precondition.h>

#include <cstdio>

namespace tt = boost::test_tools;

template <int dim>
class ExactSolution : public dealii::Function<dim>
{
public:
  ExactSolution() = default;

  double value(dealii::Point<dim> const &p,
               unsigned int const component = 0) const override;
};

template <int dim>
double ExactSolution<dim>::value(dealii::Point<dim> const &p,
                                 unsigned int const) const
{
  double val = 1.;
  for (unsigned int d = 0; d < dim; ++d)
    val *= (p[d] - 1.) * p[d];

  return val;
}

template <int dim>
class Source : public dealii::Function<dim>
{
public:
  Source() = default;

  double value(dealii::Point<dim> const &p,
               unsigned int const component = 0) const override;
};

template <int dim>
double Source<dim>::value(dealii::Point<dim> const &p, unsigned int const) const
{
  double val = 0.;
  for (unsigned int d = 0; d < dim; ++d)
  {
    double tmp = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      if (i != d)
        tmp += p[i] * (p[i] - 1.);

    val += -2. * tmp;
  }

  return val;
}

BOOST_AUTO_TEST_CASE(laplace_2d)
{
  Source<2> source;

  Laplace<2, dealii::TrilinosWrappers::MPI::Vector> laplace(MPI_COMM_WORLD, 2);
  laplace.setup_system();
  laplace.assemble_system(source);
  dealii::TrilinosWrappers::PreconditionSSOR preconditioner;
  dealii::TrilinosWrappers::MPI::Vector solution =
      laplace.solve(preconditioner);

  // The exact solution is quadratique so the error should be zero.
  ExactSolution<2> exact_solution;
  BOOST_TEST(laplace.compute_error(exact_solution), tt::tolerance(1e-14));

  laplace.output_results();
  // Remove output file
  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int world_size =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  if (rank == 0)
  {
    BOOST_TEST(std::remove("solution.pvtu") == 0);
    for (unsigned int i = 0; i < world_size; ++i)
      BOOST_TEST(
          std::remove(("solution-" + std::to_string(i) + ".vtu").c_str()) == 0);
  }
}

BOOST_AUTO_TEST_CASE(laplace_3d)
{
  Source<3> source;

  Laplace<3, dealii::TrilinosWrappers::MPI::Vector> laplace(MPI_COMM_WORLD, 2);
  laplace.setup_system();
  laplace.assemble_system(source);
  dealii::TrilinosWrappers::PreconditionSSOR preconditioner;
  dealii::TrilinosWrappers::MPI::Vector solution =
      laplace.solve(preconditioner);

  // The exact solution is quadratique so the error should be zero.
  ExactSolution<3> exact_solution;
  BOOST_TEST(laplace.compute_error(exact_solution), tt::tolerance(1e-14));
}
