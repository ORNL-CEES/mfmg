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

#ifndef MFMG_BELOS_TRAITS_HPP
#define MFMG_BELOS_TRAITS_HPP

#include <mfmg/common/exceptions.hpp>
#include <mfmg/dealii/multivector.hpp>

#include <BelosMultiVecTraits.hpp>

#include <Teuchos_RCP.hpp>

#include <vector>

// FIXME: Belos traits are only used for compilation
// In the runtime, Belos is only used in TraceMin and KrylovSchur algorithms,
// and TsqrOrthoManager

namespace Belos
{
using mfmg::NotImplementedExc;

template <typename VectorType>
class MultiVecTraits<double, mfmg::MultiVector<VectorType>>
{
  using MultiVectorType = mfmg::MultiVector<VectorType>;

public:
  static Teuchos::RCP<MultiVectorType> Clone(const MultiVectorType &mv,
                                             const int numvecs)
  {
    std::ignore = mv;
    std::ignore = numvecs;
    throw NotImplementedExc();
  }

  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv)
  {
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv,
                                                 const std::vector<int> &index)
  {
    std::ignore = mv;
    std::ignore = index;
    throw NotImplementedExc();
  }

  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv,
                                                 const Teuchos::Range1D &index)
  {
    std::ignore = mv;
    std::ignore = index;
    throw NotImplementedExc();
  }

  static Teuchos::RCP<MultiVectorType>
  CloneViewNonConst(MultiVectorType &mv, const std::vector<int> &index)
  {
    std::ignore = mv;
    std::ignore = index;
    throw NotImplementedExc();
  }

  static Teuchos::RCP<MultiVectorType>
  CloneViewNonConst(MultiVectorType &mv, const Teuchos::Range1D &index)
  {
    std::ignore = mv;
    std::ignore = index;
    throw NotImplementedExc();
  }

  static Teuchos::RCP<const MultiVectorType>
  CloneView(const MultiVectorType &mv, const std::vector<int> &index)
  {
    std::ignore = mv;
    std::ignore = index;
    throw NotImplementedExc();
  }

  static Teuchos::RCP<const MultiVectorType>
  CloneView(MultiVectorType &mv, const Teuchos::Range1D &index)
  {
    std::ignore = mv;
    std::ignore = index;
    throw NotImplementedExc();
  }

  static ptrdiff_t GetGlobalLength(const MultiVectorType &mv)
  {
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static int GetNumberVecs(const MultiVectorType &mv)
  {
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static void MvTimesMatAddMv(const double alpha, const MultiVectorType &A,
                              const Teuchos::SerialDenseMatrix<int, double> &B,
                              const double beta, MultiVectorType &mv)
  {
    std::ignore = alpha;
    std::ignore = A;
    std::ignore = B;
    std::ignore = beta;
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static void MvAddMv(const double alpha, const MultiVectorType &A,
                      const double beta, const MultiVectorType &B,
                      MultiVectorType &mv)
  {
    std::ignore = alpha;
    std::ignore = A;
    std::ignore = B;
    std::ignore = beta;
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static void MvScale(MultiVectorType &mv, const double alpha)
  {
    std::ignore = mv;
    std::ignore = alpha;
    throw NotImplementedExc();
  }

  static void MvScale(MultiVectorType &mv, const std::vector<double> &alpha)
  {
    std::ignore = mv;
    std::ignore = alpha;
    throw NotImplementedExc();
  }

  static void MvTransMv(const double alpha, const MultiVectorType &A,
                        const MultiVectorType &B,
                        Teuchos::SerialDenseMatrix<int, double> &C)
  {
    std::ignore = alpha;
    std::ignore = A;
    std::ignore = B;
    std::ignore = C;
    throw NotImplementedExc();
  }

  static void MvDot(const MultiVectorType &mv, const MultiVectorType &A,
                    std::vector<double> &b)
  {
    std::ignore = mv;
    std::ignore = A;
    std::ignore = b;
    throw NotImplementedExc();
  }

  static void
  MvNorm(const MultiVectorType &mv,
         std::vector<typename Teuchos::ScalarTraits<double>::magnitudeType>
             &normvec,
         NormType type = TwoNorm)
  {
    std::ignore = mv;
    std::ignore = normvec;
    std::ignore = type;
    throw NotImplementedExc();
  }

  static void SetBlock(const MultiVectorType &A, const std::vector<int> &index,
                       MultiVectorType &mv)
  {
    std::ignore = A;
    std::ignore = index;
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static void SetBlock(const MultiVectorType &A, const Teuchos::Range1D &index,
                       MultiVectorType &mv)
  {
    std::ignore = A;
    std::ignore = index;
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static void Assign(const MultiVectorType &A, MultiVectorType &mv)
  {
    std::ignore = A;
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static void MvRandom(MultiVectorType &mv)
  {
    std::ignore = mv;
    throw NotImplementedExc();
  }

  static void MvInit(MultiVectorType &mv, const double alpha = 0.)
  {
    std::ignore = mv;
    std::ignore = alpha;
    throw NotImplementedExc();
  }

  static void MvPrint(const MultiVectorType &mv, std::ostream &os)
  {
    std::ignore = mv;
    std::ignore = os;
    throw NotImplementedExc();
  }

  static bool HasConstantStride(const MultiVectorType &mv)
  {
    std::ignore = mv;
    throw NotImplementedExc();
  }
};

} // namespace Belos

#endif
