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

#ifndef MFMG_DEALII_OPERATOR_DEVICE_CUH
#define MFMG_DEALII_OPERATOR_DEVICE_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/concepts.hpp>
#include <mfmg/cuda_handle.cuh>
#include <mfmg/exceptions.hpp>
#include <mfmg/sparse_matrix_device.cuh>

#if MFMG_WITH_AMGX
#include <amgx_c.h>
#endif

namespace mfmg
{
template <typename VectorType>
class SparseMatrixDeviceOperator : public MatrixOperator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using matrix_type = SparseMatrixDevice<value_type>;
  using vector_type = VectorType;
  using operator_type = MatrixOperator<vector_type>;

  SparseMatrixDeviceOperator(std::shared_ptr<matrix_type> matrix);

  virtual size_t m() const override final { return _matrix->m(); }
  virtual size_t n() const override final { return _matrix->n(); }

  size_t nnz() const { return _matrix->n_nonzero_elements(); }

  virtual size_t grid_complexity() const override final { return m(); }
  virtual size_t operator_complexity() const override final { return nnz(); }

  virtual void apply(vector_type const &x, vector_type &y) const override final;

  virtual std::shared_ptr<operator_type> transpose() const override final;

  virtual std::shared_ptr<operator_type>
  multiply(operator_type const &operator_b) const override final;

  std::shared_ptr<operator_type>
  multiply_transpose(operator_type const &operator_b) const
  {
    return std::make_shared<SparseMatrixDeviceOperator<VectorType>>(nullptr);
  }

  std::shared_ptr<matrix_type> get_matrix() const { return _matrix; }

  virtual std::shared_ptr<vector_type>
  build_domain_vector() const override final;

  std::shared_ptr<vector_type> build_range_vector() const override final;

private:
  std::shared_ptr<matrix_type> _matrix;
};

template <typename VectorType>
class SmootherDeviceOperator : public Operator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using vector_type = VectorType;
  using matrix_type = SparseMatrixDevice<value_type>;
  using operator_type = Operator<vector_type>;

  SmootherDeviceOperator(matrix_type const &matrix,
                         std::shared_ptr<boost::property_tree::ptree> params);

  virtual size_t m() const override final { return _matrix.m(); }
  virtual size_t n() const override final { return _matrix.n(); }

  virtual size_t grid_complexity() const override final { return m(); }
  virtual size_t operator_complexity() const override final
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return 0;
  }

  virtual void apply(vector_type const &b, vector_type &x) const override final;

  virtual std::shared_ptr<VectorType>
  build_domain_vector() const override final;

  virtual std::shared_ptr<VectorType> build_range_vector() const override final;

private:
  void initialize(std::string &prec_type);

  matrix_type const &_matrix;

  matrix_type _smoother;
};

template <typename VectorType>
class DirectDeviceOperator : public Operator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using vector_type = VectorType;
  using matrix_type = SparseMatrixDevice<value_type>;
  using operator_type = Operator<vector_type>;

  DirectDeviceOperator(CudaHandle const &cuda_handle, matrix_type const &matrix,
                       std::shared_ptr<boost::property_tree::ptree> params);

  virtual ~DirectDeviceOperator();

  virtual size_t m() const override final { return _matrix.m(); }
  virtual size_t n() const override final { return _matrix.n(); }

  virtual size_t grid_complexity() const override final { return m(); }
  virtual size_t operator_complexity() const override final
  {
    return _matrix.n_nonzero_elements();
  }

  virtual void apply(vector_type const &b, vector_type &x) const override final;

  virtual std::shared_ptr<vector_type>
  build_domain_vector() const override final;

  virtual std::shared_ptr<vector_type>
  build_range_vector() const override final;

private:
  CudaHandle const &_cuda_handle;

  matrix_type const &_matrix;
  std::string _solver;
  std::string _amgx_config_file;
#if MFMG_WITH_AMGX
  // AMGX handles and data
  AMGX_config_handle _amgx_config_handle;
  AMGX_resources_handle _amgx_res_handle;
  AMGX_matrix_handle _amgx_matrix_handle;
  AMGX_vector_handle _amgx_rhs_handle;
  AMGX_vector_handle _amgx_solution_handle;
  AMGX_solver_handle _amgx_solver_handle;
  int _device_id[1];
  std::unordered_map<int, int> _row_map;
#endif
};
} // namespace mfmg

#endif

#endif
