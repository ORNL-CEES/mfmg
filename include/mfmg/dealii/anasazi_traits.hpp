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

#ifndef MFMG_ANASAZI_TRAITS_HPP
#define MFMG_ANASAZI_TRAITS_HPP

#include <mfmg/common/exceptions.hpp>
#include <mfmg/dealii/multivector.hpp>

#include <deal.II/lac/sparse_matrix.h>

#include <AnasaziMultiVecTraits.hpp>
#include <AnasaziOperatorTraits.hpp>

#include <algorithm>
#include <random>
#include <vector>

namespace Anasazi
{
using mfmg::ASSERT;
using mfmg::ASSERT_THROW_NOT_IMPLEMENTED;

template <typename VectorType>
class MultiVecTraits<double, mfmg::MultiVector<VectorType>>
{
  using MultiVectorType = mfmg::MultiVector<VectorType>;

public:
  static Teuchos::RCP<MultiVectorType> Clone(const MultiVectorType &mv,
                                             const int numvecs)
  {
    return Teuchos::rcp(new MultiVectorType(numvecs, mv.size()));
  }

  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv)
  {
    auto n_vectors = mv.n_vectors();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      *(*new_mv)[i] = *mv[i];

    return new_mv;
  }

  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv,
                                                 const std::vector<int> &index)
  {
    int n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      *(*new_mv)[i] = *mv[index[i]];

    return new_mv;
  }

  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv,
                                                 const Teuchos::Range1D &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      *(*new_mv)[i] = *mv[index.lbound() + i];

    return new_mv;
  }

  static Teuchos::RCP<MultiVectorType>
  CloneViewNonConst(MultiVectorType &mv, const std::vector<int> &index)
  {
    int n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      (*new_mv)[i] = mv[index[i]];

    return new_mv;
  }

  static Teuchos::RCP<MultiVectorType>
  CloneViewNonConst(MultiVectorType &mv, const Teuchos::Range1D &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      (*new_mv)[i] = mv[index.lbound() + i];

    return new_mv;
  }

  static Teuchos::RCP<const MultiVectorType>
  CloneView(const MultiVectorType &mv, const std::vector<int> &index)
  {
    int n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      (*new_mv)[i] = mv[index[i]];

    return new_mv;
  }

  static Teuchos::RCP<const MultiVectorType>
  CloneView(MultiVectorType &mv, const Teuchos::Range1D &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      (*new_mv)[i] = mv[index.lbound() + i];

    return new_mv;
  }

  static ptrdiff_t GetGlobalLength(const MultiVectorType &mv)
  {
    return mv.size();
  }

  static int GetNumberVecs(const MultiVectorType &mv) { return mv.n_vectors(); }

  static void MvTimesMatAddMv(const double alpha, const MultiVectorType &A,
                              const Teuchos::SerialDenseMatrix<int, double> &B,
                              const double beta, MultiVectorType &mv)
  {
    auto n_vectors = A.n_vectors();

    ASSERT(B.numRows() == n_vectors, "");
    ASSERT(B.numCols() == mv.n_vectors(), "");
    ASSERT(mv.size() == A.size(), "");

    for (int j = 0; j < B.numCols(); j++)
    {
      *mv[j] *= beta;
      for (int k = 0; k < n_vectors; k++)
        (*mv[j]).add(alpha * B(k, j), *A[k]);
    }
  }

  static void MvAddMv(const double alpha, const MultiVectorType &A,
                      const double beta, const MultiVectorType &B,
                      MultiVectorType &mv)
  {
    auto n_vectors = A.n_vectors();
    ASSERT(A.size() == B.size(), "");
    ASSERT(B.n_vectors() == n_vectors, "");
    ASSERT(mv.n_vectors() == B.n_vectors(), "");
    ASSERT(mv.size() == B.size(), "");

    for (int i = 0; i < n_vectors; i++)
    {
      *mv[i] = *A[i];
      (*mv[i]).sadd(alpha, beta, *B[i]);
    }
  }

  static void MvScale(MultiVectorType &mv, const double alpha)
  {
    for (int i = 0; i < mv.n_vectors(); i++)
      *mv[i] *= alpha;
  }

  static void MvScale(MultiVectorType &mv, const std::vector<double> &alpha)
  {
    auto n_vectors = mv.n_vectors();

    ASSERT(int(alpha.size()) == n_vectors, "");
    for (int i = 0; i < n_vectors; i++)
      *mv[i] *= alpha[i];
  }

  static void MvTransMv(const double alpha, const MultiVectorType &A,
                        const MultiVectorType &B,
                        Teuchos::SerialDenseMatrix<int, double> &C)
  {
    ASSERT(A.size() == B.size(), "");
    ASSERT(C.numRows() == A.n_vectors(), "");
    ASSERT(C.numCols() == B.n_vectors(), "");

    for (int i = 0; i < A.n_vectors(); i++)
      for (int j = 0; j < B.n_vectors(); j++)
        C(i, j) = alpha * (*A[i] * *B[j]);
  }

  static void MvDot(const MultiVectorType &mv, const MultiVectorType &A,
                    std::vector<double> &b)
  {
    auto n_vectors = mv.n_vectors();
    ASSERT(A.size() == mv.size(), "");
    ASSERT(A.n_vectors() == n_vectors, "");

    b.resize(n_vectors);
    for (int i = 0; i < n_vectors; i++)
    {
      b[i] = *mv[i] * *A[i];
    }
  }

  static void
  MvNorm(const MultiVectorType &mv,
         std::vector<typename Teuchos::ScalarTraits<double>::magnitudeType>
             &normvec)
  {
    auto n_vectors = mv.n_vectors();

    normvec.resize(n_vectors);
    for (int i = 0; i < n_vectors; i++)
      normvec[i] = (*mv[i]).l2_norm();
  }

  static void SetBlock(const MultiVectorType &A, const std::vector<int> &index,
                       MultiVectorType &mv)
  {
    int n_vectors = index.size();

    ASSERT(A.n_vectors() == n_vectors, "");
    ASSERT(mv.n_vectors() >= n_vectors, "");
    ASSERT(A.size() == mv.size(), "");

    for (int i = 0; i < n_vectors; i++)
      *mv[index[i]] = *A[i];
  }

  static void SetBlock(const MultiVectorType &A, const Teuchos::Range1D &index,
                       MultiVectorType &mv)
  {
    auto n_vectors = index.size();

    ASSERT(A.n_vectors() == index.size(), "");
    ASSERT(mv.n_vectors() >= index.size(), "");
    ASSERT(A.size() == mv.size(), "");

    for (int i = 0; i < n_vectors; i++)
      *mv[index.lbound() + i] = *A[i];
  }

  static void Assign(const MultiVectorType &A, MultiVectorType &mv)
  {
    std::ignore = A;
    std::ignore = mv;
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  static void MvRandom(MultiVectorType &mv)
  {
    auto n_vectors = mv.n_vectors();

    std::mt19937 gen(1337);
    std::uniform_real_distribution<double> dist(0, 1);
    for (int i = 0; i < n_vectors; i++)
    {
      auto &v = *mv[i];
      std::transform(v.begin(), v.end(), v.begin(),
                     [&](auto &) { return dist(gen); });
    }
  }

  static void MvInit(MultiVectorType &mv, const double alpha = 0.)
  {
    int n_vectors = mv.n_vectors();
    for (int i = 0; i < n_vectors; i++)
      *mv[i] = alpha;
  }

  static void MvPrint(const MultiVectorType &mv, std::ostream &os)
  {
    std::ignore = mv;
    std::ignore = os;
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
};

template <typename VectorType, typename ValueType>
class OperatorTraits<double, mfmg::MultiVector<VectorType>,
                     dealii::SparseMatrix<ValueType>>
{
  using MultiVectorType = mfmg::MultiVector<VectorType>;
  using OperatorType = dealii::SparseMatrix<ValueType>;

public:
  static void Apply(const OperatorType &op, const MultiVectorType &x,
                    MultiVectorType &y)
  {
    auto n_vectors = x.n_vectors();

    ASSERT(x.size() == y.size(), "");
    ASSERT(y.n_vectors() == n_vectors, "");

    for (int i = 0; i < n_vectors; i++)
      op.vmult(*y[i], *x[i]);
  };
};

} // namespace Anasazi

#endif
