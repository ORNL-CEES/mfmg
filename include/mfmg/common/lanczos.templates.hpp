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

#ifndef MFMG_LANCZOS_LANCZOS_TEMPLATE_HPP
#define MFMG_LANCZOS_LANCZOS_TEMPLATE_HPP

#include <mfmg/cuda/utils.cuh>

#include <algorithm>
#include <random>
#include <vector>

#include "lanczos.hpp"
#include "lanczos_deflatedop.templates.hpp"

// This complex code has to be included before lapacke for the code to compile.
// Otherwise, it conflicts with boost or Kokkos.
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>

namespace mfmg
{

namespace internal
{
template <typename VectorType>
void details_set_initial_guess(VectorType &initial_guess, int seed)
{
  // Modify initial guess with a random noise by multiplying each entry of the
  // vector with a random value from a uniform distribution. This specific
  // procedure guarantees that zero entries of the vector stay zero, which is
  // important for situations where they are associated with constrained dofs in
  // Deal.II
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0, 1);
  std::transform(initial_guess.begin(), initial_guess.end(),
                 initial_guess.begin(),
                 [&](auto &v) { return (1. + dist(gen)) * v; });
}

#ifdef __CUDACC__
template <>
void details_set_initial_guess(
    dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA> &initial_guess,
    int seed)
{
  // Modify initial guess with a random noise by multiplying each entry of the
  // vector with a random value from a uniform distribution. This specific
  // procedure guarantees that zero entries of the vector stay zero, which is
  // important for situations where they are associated with constrained dofs in
  // Deal.II
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0, 1);
  std::vector<double> initial_guess_host(initial_guess.local_size());
  cuda_mem_copy_to_host(initial_guess.get_values(), initial_guess_host);
  std::transform(initial_guess_host.begin(), initial_guess_host.end(),
                 initial_guess_host.begin(),
                 [&](auto &v) { return (1. + dist(gen)) * v; });
  cuda_mem_copy_to_dev(initial_guess_host, initial_guess.get_values());
}
#endif
} // namespace internal

/// \brief Lanczos solver: constructor
template <typename OperatorType, typename VectorType>
Lanczos<OperatorType, VectorType>::Lanczos(OperatorType const &op) : _op(op)
{
  ASSERT(_op.m() == _op.n(), "Operator must be square");
}

/// \brief Lanczos solver: perform Lanczos solve, use random initial guess
template <typename OperatorType, typename VectorType>
std::tuple<std::vector<double>, std::vector<VectorType>>
Lanczos<OperatorType, VectorType>::solve(
    boost::property_tree::ptree const &params, VectorType initial_guess) const
{
  bool const is_deflated = params.get<bool>("is_deflated", false);

  int const n_eigenvectors = params.get<int>("num_eigenpairs");

  int num_cycles = 1;
  int num_evecs_per_cycle = n_eigenvectors;
  if (is_deflated)
  {
    num_evecs_per_cycle = params.get<int>("num_eigenpairs_per_cycle");
    num_cycles = params.get<int>("num_cycles");
  }

  ASSERT(num_cycles >= 1, "Number of cycles must be positive");
  ASSERT(num_evecs_per_cycle >= 1,
         "Number of computed eigenpairs per cycle must be positive");

  std::vector<double> evals;
  std::vector<VectorType> evecs;

  // Form deflated operator from original operator.
  // NOTE: for regular Lanczos, it will never do any deflation
  DeflatedOperator<OperatorType, VectorType> deflated_op(_op);

  // Loop over Lanczos solves
  for (int cycle = 0; cycle < num_cycles; ++cycle)
  {
    // For the first cycle, we use the untouched user-provided initial guess.
    // This is consistent with what's done in ARPACK and this avoid any bad
    // surprise where the initial guess provided is different than the initial
    // guess used.
    if (cycle > 0)
      internal::details_set_initial_guess(initial_guess, cycle);

    deflated_op.deflate(initial_guess);

    std::vector<double> cycle_evals;
    std::vector<VectorType> cycle_evecs;
    std::tie(cycle_evals, cycle_evecs) = details_solve_lanczos(
        deflated_op, num_evecs_per_cycle, params, initial_guess);

    // Save the eigenpairs just calculated

    // NOTE: throughout we use the term eigenpair (= eigenvector,
    // eigenvalue), though the precise terminology should be "approximate
    // eigenpairs" or "Ritz pairs."
    for (int i = 0; i < num_evecs_per_cycle; ++i)
    {
      evals.push_back(cycle_evals[i]);
      evecs.push_back(cycle_evecs[i]);
    }

    // Add eigenvectors to the set of vectors being deflated out
    // Do not deflate last cycle. For the regular Lanczos this means do not
    // deflate ever.
    if (cycle != num_cycles - 1)
    {
      deflated_op.add_deflation_vecs(cycle_evecs);
    }
  }

  // Squeeze if num_cycles * num_eigenpairs_per_cycle > n_eigenvectors
  evals.resize(n_eigenvectors);
  evecs.resize(n_eigenvectors);

  // Sort eigenvalues in ascending order if we run multiple Lanczos
  // The reason is that while each Lanczos solve produces sorted eigenvalues,
  // they may be the same, and thus need to be interleaved
  if (num_cycles > 1)
  {
    std::vector<double> sorted_evals(n_eigenvectors);
    std::vector<VectorType> sorted_evecs(n_eigenvectors);

    // Compute permutation for ascending order
    std::vector<int> perm_index(n_eigenvectors);
    std::iota(perm_index.begin(), perm_index.end(), 0);
    std::sort(perm_index.begin(), perm_index.end(),
              [&](int i, int j) { return evals[i] < evals[j]; });

    for (int i = 0; i < n_eigenvectors; i++)
    {
      sorted_evals[i] = evals[perm_index[i]];
      sorted_evecs[i] = evecs[perm_index[i]];
    }
    evals = sorted_evals;
    evecs = sorted_evecs;
  }

  return std::make_tuple(evals, evecs);
}

/// \brief Lanczos solver: perform Lanczos solve
template <typename OperatorType, typename VectorType>
template <typename FullOperatorType>
std::tuple<std::vector<double>, std::vector<VectorType>>
Lanczos<OperatorType, VectorType>::details_solve_lanczos(
    FullOperatorType const &op, int const num_requested,
    boost::property_tree::ptree const &params, VectorType const &initial_guess)
{
  int const maxit = params.get<int>("max_iterations");
  double const tol = params.get<double>("tolerance");
  int const percent_overshoot = params.get<int>("percent_overshoot", 0);

  ASSERT(0 <= percent_overshoot && percent_overshoot < 100,
         "Lanczos overshoot percentage should be in [0, 100)");
  ASSERT(tol >= 0., "Lanczos tolerance must be non-negative");
  ASSERT(maxit >= num_requested, "Lanczos max iterations is too small to "
                                 "produce required number of eigenvectors.");

  int const n = op.n();

  std::vector<double> evals;
  std::vector<VectorType> evecs;

  // Initializations; first Lanczos vector.
  double alpha = 0;
  double beta = initial_guess.l2_norm();

  std::vector<VectorType> lanc_vectors; // Lanczos vectors

  // Create first Lanczos vector if necessary
  if (lanc_vectors.size() < 1)
    lanc_vectors.push_back(initial_guess);

  std::vector<double> main_diagonal;
  std::vector<double> sub_diagonal;

  std::vector<double>
      evecs_tridiag; // eigenvectors of tridiagonal matrix, stored in flat array

  // Lanczos iteration loop
  int it = 1;
  for (int it_prev_check = 0; it <= maxit; ++it)
  {
    // Normalize lanczos vector
    ASSERT(beta, "Internal error"); // TODO: set up better check for near-zero

    lanc_vectors[it - 1] /= beta;

    if (lanc_vectors.size() < static_cast<size_t>(it + 1))
    {
      // Add new Lanczos vector
      lanc_vectors.push_back(VectorType(n));
    }

    // Apply operator.
    op.vmult(lanc_vectors[it], lanc_vectors[it - 1]);

    // Compute, apply, save Lanczos coefficients
    if (it != 1)
    {
      lanc_vectors[it].add(-beta, lanc_vectors[it - 2]);
      sub_diagonal.push_back(beta);
    }

    alpha = lanc_vectors[it - 1] * lanc_vectors[it]; // = tridiag_{it,it}

    main_diagonal.push_back(alpha);

    lanc_vectors[it].add(-alpha, lanc_vectors[it - 1]);

    beta = lanc_vectors[it].l2_norm(); // = tridiag_{it+1,it}

    // Check convergence if requested
    // NOTE: an alternative here for p > 0 is
    // int((100./p)*ln(it)) > int((100./p)*ln(it-1))
    bool const first_iteration = (it == 1);
    bool const max_iterations_reached = (it == maxit);
    bool const percent_overshoot_exceeded =
        (100 * (it - it_prev_check) > percent_overshoot * it_prev_check);
    if (first_iteration || max_iterations_reached || percent_overshoot_exceeded)
    {
      int const dim_hessenberg = it;
      ASSERT((size_t)dim_hessenberg == main_diagonal.size(), "Internal error");

      // Calculate eigenpairs of tridiagonal matrix for convergence test or at
      // last iteration
      std::tie(evals, evecs_tridiag) = details_calc_tridiag_epairs(
          main_diagonal, sub_diagonal, num_requested);

      if (details_check_convergence(beta, dim_hessenberg, num_requested, tol,
                                    evecs_tridiag))
      {
        break;
      }

      // Record iteration number when this check done
      it_prev_check = it;
    }
  }
  it = it < maxit ? it : maxit;

  ASSERT(it >= num_requested,
         "Internal error: required number of iterations not reached");

  // Calculate full operator eigenvectors from tridiagonal eigenvectors.
  // ISSUE: may be needed to modify this code to not save all Lanczos vectors
  // but instead recalculate them for this use.
  // However this may be dangerous if the second Lanczos iteration
  // has different roundoff characteristics, e.g., due to order of
  // operation differences.
  // ISSUE: we have not taken precautions here with regard to
  // potential impacts of loss of orthogonality of Lanczos vectors.
  evecs = details_calc_evecs(num_requested, it, lanc_vectors, evecs_tridiag);

  return std::make_tuple(evals, evecs);
}

/// \brief Lanczos solver: calculate eigenpairs from tridiagonal of Lanczos
/// coefficients
template <typename OperatorType, typename VectorType>
std::tuple<std::vector<double>, std::vector<double>>
Lanczos<OperatorType, VectorType>::details_calc_tridiag_epairs(
    std::vector<double> const &main_diagonal,
    std::vector<double> const &sub_diagonal, int const num_requested)
{
  int const n = main_diagonal.size();

  ASSERT(n >= 1, "Internal error: tridigonal matrix size must be positive");
  ASSERT(sub_diagonal.size() == (size_t)(n - 1),
         "Internal error: mismatch in main and off-diagonal sizes");

  std::vector<double> evals;
  std::vector<double> evecs; // flat array

  if (n < num_requested)
  {
    return std::make_tuple(evals, evecs);
  }

  // dstyev destroys storage
  evals = main_diagonal;
  std::vector<double> sub_diagonal_aux = sub_diagonal;
  std::vector<double> evecs_aux(n * n);

  // As the matrix is symmetric and tridiagonal, we use DSTEV LAPACK routine,
  // which computes all eigenvalues and, optionally, eigenvectors of a real
  // symmetric tridiagonal matrix A.
  //   http://www.netlib.org/lapack/explore-html/d7/d48/dstev_8f.html
  // It guarantees that the eigenvalues are returned in ascending order.
  // NOTE: as some arguments are invalidated during the routine, we make a copy
  // of the off-diagonal prior to the call.
  lapack_int const info =
      LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', n, evals.data(),
                    sub_diagonal_aux.data(), evecs_aux.data(), n);
  ASSERT(!info, "Call to LAPACKE_dstev failed.");

  // The following approach is taken from Cullum and Willoughby vol. 1, to
  // identify and remove spurious and redundant eigenvalues. Here, one computes
  // the eigenvalues of the tridiagonal matrix T and also the matrix T2 formed
  // by removing the first row and col of T. If an eigenvalue of T is not
  // repeated, and it is also an eigenvalue of T2, then it is considered
  // spurious.
  //
  // The C/W algorithm has other nuances which may necessitate more
  // revisions of the simplified algorithm implemented here.

  // The following tolerance may need adjustment, however this doesn't
  // seem critical (cf. C/W vol. 1). Cullum/Willoughby use 1e-12.
  double const tol2 = 5.e-12;

  // Identify repeated eigenvalues.
  // A "marked" eigenvalue here is nonrepeated or first of a set of repeated.
  std::vector<bool> is_repeated(n);
  std::vector<bool> is_marked(n);
  for (int i = 0; i < n; ++i)
  {
    // A Ritz value is considered repeated if it is within tolerance tol2 of
    // any other value. As evals are sorted, it is sufficient to check whether
    // the value is within tol2 of the previous and next values.
    is_repeated[i] = ((i > 0 && (evals[i] <= evals[i - 1] + tol2)) ||
                      (i < n - 1 && (evals[i + 1] <= evals[i] + tol2)));

    // A Ritz value is marked (i.e., a good value) if it is either more than
    // tol2 distance away from all other values, or it is the first one of the
    // repeated values.
    is_marked[i] = (i == 0 || (evals[i] > evals[i - 1] + tol2));
  }

  // Identify spurious eigenvalues
  std::vector<bool> is_spurious(n, false);
  int const n2 = n - 1;
  if (n2 >= 1 && n2 >= num_requested)
  {
    // Solve an eigenproblem based on deleting the first row and col of the
    // original tridiag matrix.
    // NOTE: as we only need eigenvalues and not eigenvectors, a different
    // LAPACK call may be more appropriate.
    std::vector<double> evals2(++main_diagonal.begin(), main_diagonal.end());
    std::vector<double> sub_diagonal_aux2(++sub_diagonal.begin(),
                                          sub_diagonal.end());
    std::vector<double> evecs_aux2(n2 * n2);

    lapack_int const info2 =
        LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', n2, evals2.data(),
                      sub_diagonal_aux2.data(), evecs_aux2.data(), n2);
    ASSERT(!info2, "Call to LAPACKE_dstev failed.");

    // Loop over original eigenvalues to check for spuriousness.
    // NOTE: assuming here evals and evals2 are in ascending order.
    int j_start = 0;
    for (int i = 0; i < n; ++i)
    {
      if (is_repeated[i])
      {
        // A repeated eigenvalue of T is never spurious.
        continue;
      }

      // Seek matching T2 eigenvalue. If found, then evals[i] is spurious.
      // Note the looping is rigged here to avoid an O(n^2) algorithm.
      for (int j = j_start; j < n2; ++j)
      {
        bool const is_t2j_below = (evals2[j] < evals[i] - tol2);
        if (is_t2j_below)
        {
          // For the next i, evals[i] will be >= this one,
          // so not examining this j for next i will be ok.
          j_start = j;
          continue;
        }

        bool const is_t2j_above = (evals2[j] > evals[i] + tol2);
        if (is_t2j_above)
          // we have passed up evals[i], so no match.
          break;

        // in interval surrounding evals[i], thus evals[i] spurious.
        is_spurious[i] = true;
        break;
      }
    }
  }

  // Purge spurious and redundant eigenvalues and corresponding eigenvectors
  int num_available = n;
  for (int i = n - 1; i >= 0; --i)
  {
    if (is_spurious[i] || !is_marked[i])
    {
      evals.erase(evals.begin() + i);
      evecs_aux.erase(evecs_aux.begin() + n * i,
                      evecs_aux.begin() + n * (i + 1));
      num_available--;
    }
  }

  if (num_available < num_requested)
  {
    return std::make_tuple(std::vector<double>(), std::vector<double>());
  }

  // Save results.
  evals.resize(num_requested);
  evecs.resize(n * num_requested);
  for (int i = 0; i < num_requested; ++i)
  {
    auto first = evecs.begin() + n * i;
    auto aux_first = evecs_aux.begin() + n * i;
    auto aux_last = aux_first + n;
    double const norm =
        std::sqrt(std::inner_product(aux_first, aux_last, aux_first, 0.));
    std::transform(aux_first, aux_last, first,
                   [norm](auto &v) { return v / norm; });
  }
  return std::make_tuple(evals, evecs);
}

/// \brief Lanczos solver: perform convergence check
template <typename OperatorType, typename VectorType>
bool Lanczos<OperatorType, VectorType>::details_check_convergence(
    double beta, int const num_evecs, int const num_requested, double tol,
    std::vector<double> const &evecs)
{
  // Must iterate at least until we have num_requested eigenpairs
  if (num_evecs < num_requested)
    return false;

  bool is_converged = true;

  // Terminate if every approximate eigenvalue has converged to tolerance
  // ISSUE: here ignoring possible nuances regarding the correctness
  // of this check.
  // NOTE: k may be desirable to "scale" the stopping criterion
  // based on (estimate of) matrix norm or similar.
  for (int i = 0; i < num_requested; ++i)
  {
    double const bound = beta * std::abs(evecs[num_evecs - 1 + num_evecs * i]);
    is_converged = is_converged && bound <= tol;
  }

  return is_converged;
}

/// \brief Lanczos solver: calculate full (approx) eigenvectors from tridiag
/// eigenvectors
template <typename OperatorType, typename VectorType>
std::vector<VectorType> Lanczos<OperatorType, VectorType>::details_calc_evecs(
    int const num_requested, int const n,
    std::vector<VectorType> const &lanc_vectors,
    std::vector<double> const &evecs_tridiag)
{
  auto dim = lanc_vectors[0].size();

  std::vector<VectorType> evecs(num_requested, VectorType(dim));

  // Matrix-matrix product to convert tridiagonal eigenvectors to operator
  // eigenvectors
  for (int i = 0; i < num_requested; ++i)
  {
    evecs[i] = 0.0;
    for (int j = 0; j < n; ++j)
      evecs[i].add(evecs_tridiag[j + n * i], lanc_vectors[j]);
  }

  return evecs;
}

} // namespace mfmg

#endif
