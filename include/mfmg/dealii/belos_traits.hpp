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

#include "multivector.hpp"
#include <BelosMultiVecTraits.hpp>

// FIXME: Belos traits are only used for compilation
// In the runtime, Belos is only used in TraceMin and KrylovSchur algorithms,
// and TsqrOrthoManager

namespace Belos
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
    std::ignore = mv;
    std::ignore = numvecs;
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Creates a new \c MV and copies contents of \c mv into the new
    vector (deep copy).

    \return Reference-counted pointer to the new multivector of type \c MV.
  */
  static Teuchos::RCP<MultiVectorType> CloneCopy(const MultiVectorType &mv)
  {
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
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
    std::ignore = mv;
    std::ignore = index;
    throw std::runtime_error("Not implemented");
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
    std::ignore = mv;
    std::ignore = index;
    throw std::runtime_error("Not implemented");
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
    std::ignore = mv;
    std::ignore = index;
    throw std::runtime_error("Not implemented");
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
    std::ignore = mv;
    std::ignore = index;
    throw std::runtime_error("Not implemented");
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
    std::ignore = mv;
    std::ignore = index;
    throw std::runtime_error("Not implemented");
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
    std::ignore = mv;
    std::ignore = index;
    throw std::runtime_error("Not implemented");
  }

  /// Return the number of rows in the given multivector \c mv.
  static ptrdiff_t GetGlobalLength(const MultiVectorType &mv)
  {
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
  }

  //! Obtain the number of vectors in \c mv
  static int GetNumberVecs(const MultiVectorType &mv)
  {
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Update \c mv with \f$ \alpha AB + \beta mv \f$.
   */
  static void MvTimesMatAddMv(const double alpha, const MultiVectorType &A,
                              const Teuchos::SerialDenseMatrix<int, double> &B,
                              const double beta, MultiVectorType &mv)
  {
    std::ignore = alpha;
    std::ignore = A;
    std::ignore = B;
    std::ignore = beta;
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Replace \c mv with \f$\alpha A + \beta B\f$.
   */
  static void MvAddMv(const double alpha, const MultiVectorType &A,
                      const double beta, const MultiVectorType &B,
                      MultiVectorType &mv)
  {
    std::ignore = alpha;
    std::ignore = A;
    std::ignore = B;
    std::ignore = beta;
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
  }

  static void MvScale(MultiVectorType &mv, const double alpha)
  {
    std::ignore = mv;
    std::ignore = alpha;
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Scale each element of the vectors in \c mv with \c alpha.
   */
  static void MvScale(MultiVectorType &mv, const std::vector<double> &alpha)
  {
    std::ignore = mv;
    std::ignore = alpha;
    throw std::runtime_error("Not implemented");
  }

  /// \brief Compute <tt>C := alpha * A^H B</tt>.
  ///
  /// The result C is a dense, globally replicated matrix.
  static void MvTransMv(const double alpha, const MultiVectorType &A,
                        const MultiVectorType &B,
                        Teuchos::SerialDenseMatrix<int, double> &C)
  {
    std::ignore = alpha;
    std::ignore = A;
    std::ignore = B;
    std::ignore = C;
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Compute a vector \c b where the components are the individual
   * dot-products of the \c i-th columns of \c A and \c mv, i.e.\f$b[i] =
   * A[i]^Hmv[i]\f$.
   */
  static void MvDot(const MultiVectorType &mv, const MultiVectorType &A,
                    std::vector<double> &b)
  {
    std::ignore = mv;
    std::ignore = A;
    std::ignore = b;
    throw std::runtime_error("Not implemented");
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
             &normvec,
         NormType type = TwoNorm)
  {
    std::ignore = mv;
    std::ignore = normvec;
    std::ignore = type;
    throw std::runtime_error("Not implemented");
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
    std::ignore = A;
    std::ignore = index;
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
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
    std::ignore = A;
    std::ignore = index;
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
  }

  /// \brief mv := A
  ///
  /// Assign (deep copy) A into mv.
  static void Assign(const MultiVectorType &A, MultiVectorType &mv)
  {
    std::ignore = A;
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Replace the vectors in \c mv with random vectors.
   */
  static void MvRandom(MultiVectorType &mv)
  {
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Replace each element of the vectors in \c mv with \c alpha.
   */
  static void MvInit(MultiVectorType &mv, const double alpha = 0.)
  {
    std::ignore = mv;
    std::ignore = alpha;
    throw std::runtime_error("Not implemented");
  }

  /*! \brief Print the \c mv multi-vector to the \c os output stream.
   */
  static void MvPrint(const MultiVectorType &mv, std::ostream &os)
  {
    std::ignore = mv;
    std::ignore = os;
    throw std::runtime_error("Not implemented");
  }

  static bool HasConstantStride(const MultiVectorType &mv)
  {
    std::ignore = mv;
    throw std::runtime_error("Not implemented");
  }
};

} // namespace Belos

#endif
