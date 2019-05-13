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

#include <deal.II/lac/sparse_matrix.h>

#include <algorithm>
#include <random>

#include "multivector.hpp"
#include <AnasaziMultiVecTraits.hpp>
#include <AnasaziOperatorTraits.hpp>

namespace Anasazi
{
using mfmg::ASSERT;

template <typename VectorType>
class MultiVecTraits<double, mfmg::MultiVector<VectorType>>
{
  using MultiVectorType = mfmg::MultiVector<VectorType>;

public:
  /*! \brief Creates a new empty \c MV containing \c numvecs columns.

  \return Reference-counted pointer to the new multivector of type \c MV.
  */
  static Teuchos::RCP<MultiVectorType> Clone(const MultiVectorType &mv,
                                             const int numvecs)
  {
    return Teuchos::rcp(new MultiVectorType(numvecs, mv.size()));
  }

  /*! \brief Creates a new \c MV and copies contents of \c mv into the new
    vector (deep copy).

    \return Reference-counted pointer to the new multivector of type \c MV.
  */
  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv)
  {
    auto n_vectors = mv.n_vectors();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      *(*new_mv)[i] = *mv[i];

    return new_mv;
  }

  /*! \brief Creates a new \c MV and copies the selected contents of \c mv into
    the new vector (deep copy).

    The copied vectors from \c mv are indicated by the \c index.size() indices
    in \c index. \return Reference-counted pointer to the new multivector of
    type \c MV.
  */
  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv,
                                                 const std::vector<int> &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      *(*new_mv)[i] = *mv[index[i]];

    return new_mv;
  }

  /// \brief Deep copy of specified columns of mv
  ///
  /// Create a new MV, and copy (deep copy) the columns of mv
  /// specified by the given inclusive index range into the new
  /// multivector.
  ///
  /// \param mv [in] Multivector to copy
  /// \param index [in] Inclusive index range of columns of mv
  /// \return Reference-counted pointer to the new multivector of type \c MV.
  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv,
                                                 const Teuchos::Range1D &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      *(*new_mv)[i] = *mv[index.lbound() + i];

    return new_mv;
  }

  /*! \brief Creates a new \c MV that shares the selected contents of \c mv
  (shallow copy).

  The index of the \c numvecs vectors shallow copied from \c mv are indicated by
  the indices given in \c index. \return Reference-counted pointer to the new
  multivector of type \c MV.
  */
  static Teuchos::RCP<MultiVectorType>
  CloneViewNonConst(MultiVectorType &mv, const std::vector<int> &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      (*new_mv)[i] = mv[index[i]];

    return new_mv;
  }

  /// \brief Non-const view of specified columns of mv
  ///
  /// Return a non-const view of the columns of mv specified by the
  /// given inclusive index range.
  ///
  /// \param mv [in] Multivector to view (shallow non-const copy)
  /// \param index [in] Inclusive index range of columns of mv
  /// \return Reference-counted pointer to the non-const view of specified
  /// columns of mv
  static Teuchos::RCP<MultiVectorType>
  CloneViewNonConst(MultiVectorType &mv, const Teuchos::Range1D &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      (*new_mv)[i] = mv[index.lbound() + i];

    return new_mv;
  }

  /*! \brief Creates a new const \c MV that shares the selected contents of \c
  mv (shallow copy).

  The index of the \c numvecs vectors shallow copied from \c mv are indicated by
  the indices given in \c index. \return Reference-counted pointer to the new
  const multivector of type \c MV.
  */
  static Teuchos::RCP<const MultiVectorType>
  CloneView(const MultiVectorType &mv, const std::vector<int> &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      (*new_mv)[i] = mv[index[i]];

    return new_mv;
  }

  /// \brief Const view of specified columns of mv
  ///
  /// Return a const view of the columns of mv specified by the
  /// given inclusive index range.
  ///
  /// \param mv [in] Multivector to view (shallow const copy)
  /// \param index [in] Inclusive index range of columns of mv
  /// \return Reference-counted pointer to the const view of specified columns
  /// of mv
  static Teuchos::RCP<const MultiVectorType>
  CloneView(MultiVectorType &mv, const Teuchos::Range1D &index)
  {
    auto n_vectors = index.size();

    auto new_mv = Teuchos::rcp(new MultiVectorType(n_vectors));
    for (int i = 0; i < n_vectors; i++)
      (*new_mv)[i] = mv[index.lbound() + i];

    return new_mv;
  }

  /// Return the number of rows in the given multivector \c mv.
  static ptrdiff_t GetGlobalLength(const MultiVectorType &mv)
  {
    return mv.size();
  }

  //! Obtain the number of vectors in \c mv
  static int GetNumberVecs(const MultiVectorType &mv) { return mv.n_vectors(); }

  /*! \brief Update \c mv with \f$ \alpha AB + \beta mv \f$.
   */
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

  /*! \brief Replace \c mv with \f$\alpha A + \beta B\f$.
   */
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

  /*! \brief Scale each element of the vectors in \c mv with \c alpha.
   */
  static void MvScale(MultiVectorType &mv, const std::vector<double> &alpha)
  {
    auto n_vectors = mv.n_vectors();

    ASSERT(alpha.size() == n_vectors, "");
    for (int i = 0; i < n_vectors; i++)
      *mv[i] *= alpha[i];
  }

  /// \brief Compute <tt>C := alpha * A^H B</tt>.
  ///
  /// The result C is a dense, globally replicated matrix.
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

  /*! \brief Compute a vector \c b where the components are the individual
   * dot-products of the \c i-th columns of \c A and \c mv, i.e.\f$b[i] =
   * A[i]^Hmv[i]\f$.
   */
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

  //@}
  //! @name Norm method
  //@{

  /*! \brief Compute the 2-norm of each individual vector of \c mv.
    Upon return, \c normvec[i] holds the value of \f$||mv_i||_2\f$, the \c i-th
    column of \c mv.
  */
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

  //@}

  //! @name Initialization methods
  //@{
  /*! \brief Copy the vectors in \c A to a set of vectors in \c mv indicated by
  the indices given in \c index.

  The \c numvecs vectors in \c A are copied to a subset of vectors in \c mv
  indicated by the indices given in \c index, i.e.<tt> mv[index[i]] = A[i]</tt>.
  */
  static void SetBlock(const MultiVectorType &A, const std::vector<int> &index,
                       MultiVectorType &mv)
  {
    auto n_vectors = index.size();

    ASSERT(A.n_vectors() == index.size(), "");
    ASSERT(mv.n_vectors() >= index.size(), "");
    ASSERT(A.size() == mv.size(), "");

    for (int i = 0; i < n_vectors; i++)
      *mv[index[i]] = *A[i];
  }

  /// \brief Deep copy of A into specified columns of mv
  ///
  /// (Deeply) copy the first <tt>index.size()</tt> columns of \c A
  /// into the columns of \c mv specified by the given index range.
  ///
  /// Postcondition: <tt>mv[i] = A[i - index.lbound()]</tt>
  /// for all <tt>i</tt> in <tt>[index.lbound(), index.ubound()]</tt>
  ///
  /// \param A [in] Source multivector
  /// \param index [in] Inclusive index range of columns of mv;
  ///   index set of the target
  /// \param mv [out] Target multivector
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

  /// \brief mv := A
  ///
  /// Assign (deep copy) A into mv.
  static void Assign(const MultiVectorType &A, MultiVectorType &mv)
  {
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Replace the vectors in \c mv with random vectors.
   */
  static void MvRandom(MultiVectorType &mv)
  {
    auto n_vectors = mv.n_vectors();

    std::mt19937 gen(1337);
    std::uniform_real_distribution<double> dist(0, 1);
    for (int i = 0; i < n_vectors; i++)
    {
      auto &v = *mv[i];
      std::transform(v.begin(), v.end(), v.begin(),
                     [&](auto &x) { return dist(gen); });
    }
  }

  /*! \brief Replace each element of the vectors in \c mv with \c alpha.
   */
  static void MvInit(MultiVectorType &mv, const double alpha = 0.)
  {
    auto n_vectors = mv.n_vectors();
    for (int i = 0; i < mv.n_vectors(); i++)
      *mv[i] = alpha;
  }

  /*! \brief Print the \c mv multi-vector to the \c os output stream.
   */
  static void MvPrint(const MultiVectorType &mv, std::ostream &os)
  {
    throw std::runtime_error("Not implemented");
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
