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

#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

#include <csignal>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#if MFMG_WITH_STACKTRACE
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>
#endif

namespace mfmg
{
#if MFMG_DEBUG
#if MFMG_WITH_STACKTRACE
inline void signal_handler(int signal)
{
  if (signal == SIGABRT)
  {
    std::cerr << boost::stacktrace::stacktrace();
  }
  else
  {
    std::cerr << "Unexpected signal " << signal << " received\n";
  }
  std::_Exit(EXIT_FAILURE);
}
#endif

inline void ASSERT(bool cond, std::string const &message)
{
  if (!(cond))
  {
#if MFMG_WITH_STACKTRACE
    std::signal(SIGABRT, signal_handler);
#endif
    std::cerr << message << std::endl;
    std::abort();
  }
}
#else
inline void ASSERT(bool, std::string const &) {}
#endif

inline void ASSERT_THROW(bool cond, std::string const &message)
{
  if (cond == false)
    throw std::runtime_error(message);
}

class NotImplementedExc : public std::exception
{
public:
  virtual ~NotImplementedExc() override = default;

private:
  virtual const char *what() const throw() final override
  {
    return "The function is not implemented";
  }
};

inline void ASSERT_THROW_NOT_IMPLEMENTED()
{
#if MFMG_WITH_STACKTRACE
  std::cerr << boost::stacktrace::stacktrace();
#endif
  NotImplementedExc exception;
  throw exception;
}
} // namespace mfmg

#ifdef __CUDACC__
#include <cusolverDn.h>
#include <cusparse.h>

#if MFMG_WITH_CUDA

namespace mfmg
{
#if MFMG_DEBUG
namespace internal
{
// cuSPARSE does not have a function similar to cudaGetErrorString so we
// implement our own
inline std::string get_cusparse_error_string(cusparseStatus_t error_code)
{
  switch (error_code)
  {
  case CUSPARSE_STATUS_NOT_INITIALIZED:
  {
    return "The cuSPARSE library was not initialized";
  }
  case CUSPARSE_STATUS_ALLOC_FAILED:
  {
    return "Resource allocation failed inside the cuSPARSE library";
  }
  case CUSPARSE_STATUS_INVALID_VALUE:
  {
    return "An unsupported value of parameter was passed to the function";
  }
  case CUSPARSE_STATUS_ARCH_MISMATCH:
  {
    return "The function requires a feature absent from the device "
           "architecture";
  }
  case CUSPARSE_STATUS_MAPPING_ERROR:
  {
    return "An access to GPU memory space failed";
  }
  case CUSPARSE_STATUS_EXECUTION_FAILED:
  {
    return "The GPU program failed to execute";
  }
  case CUSPARSE_STATUS_INTERNAL_ERROR:
  {
    return "An internal cuSPARSE operation failed";
  }
  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
  {
    return "The matrix type is not supported by this function";
  }
  default:
  {
    return "Unknown error";
  }
  }
}

// cuSolver does not have a function similar to cudaGetErrorString so we
// implement our own
inline std::string get_cusolver_error_string(cusolverStatus_t error_code)
{
  std::string message;
  switch (error_code)
  {
  case CUSOLVER_STATUS_NOT_INITIALIZED:
  {
    message = "The cuSolver library was not initialized";

    break;
  }
  case CUSOLVER_STATUS_ALLOC_FAILED:
  {
    message = "Resource allocation failed inside the cuSolver library";

    break;
  }
  case CUSOLVER_STATUS_INVALID_VALUE:
  {
    message = "An unsupported value of parameter was passed to the function";

    break;
  }
  case CUSOLVER_STATUS_ARCH_MISMATCH:
  {
    message =
        "The function requires a feature absent from the device architecture";

    break;
  }
  case CUSOLVER_STATUS_EXECUTION_FAILED:
  {
    message = "The GPU program failed to execute";

    break;
  }
  case CUSOLVER_STATUS_INTERNAL_ERROR:
  {
    message = "An internal cuSolver operation failed";

    break;
  }
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
  {
    message = "The matrix type is not supported by this function";

    break;
  }
  default:
  {
    message = "Unknown error";
  }
  }

  return message;
}
} // namespace internal

inline void ASSERT_CUDA(cudaError_t error_code)
{
  std::string message = cudaGetErrorString(error_code);
  ASSERT(error_code == cudaSuccess, "CUDA error: " + message);
}

inline void ASSERT_CUDA_SYNCHRONIZE() { ASSERT_CUDA(cudaDeviceSynchronize()); }

inline void ASSERT_CUSPARSE(cusparseStatus_t error_code)
{
  std::string message = internal::get_cusparse_error_string(error_code);
  ASSERT(error_code == CUSPARSE_STATUS_SUCCESS, "cuSPARSE error: " + message);
}

inline void ASSERT_CUSOLVER(cusolverStatus_t error_code)
{
  std::string message = internal::get_cusolver_error_string(error_code);
  ASSERT(error_code == CUSOLVER_STATUS_SUCCESS, "cuSolver error: " + message);
}
#else  // #if MFMG_DEBUG
inline void ASSERT_CUDA(cudaError_t error_code) { (void)error_code; }
inline void ASSERT_CUDA_SYNCHRONIZE() {}
inline void ASSERT_CUSPARSE(cusparseStatus_t error_code) { (void)error_code; }
inline void ASSERT_CUSOLVER(cusolverStatus_t error_code) { (void)error_code; }
#endif // #if MFMG_DEBUG

} // namespace mfmg
#endif // #if MFMG_WITH_CUDA

#endif // #ifdef __CUDACC__

#endif
