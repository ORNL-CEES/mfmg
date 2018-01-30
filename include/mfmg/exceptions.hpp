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

#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

#include <exception>
#include <stdexcept>
#include <string>

namespace mfmg
{
#if MFMG_DEBUG
inline void ASSERT(bool cond, std::string const &message)
{
  if (!(cond))
  {
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
  const char *what() const throw() final
  {
    return "The function is not implemented";
  }
};

inline void ASSERT_THROW_NOT_IMPLEMENTED()
{
  NotImplementedExc exception;
  throw exception;
}

#if MFMG_WITH_CUDA
#if MFMG_DEBUG
inline void ASSERT_CUDA(cudaError_t error_code)
{
  std::string message = cudaGetErrorString(error_code);
  ASSERT(error_code == cudaSuccess, "CUDA error: " + message);
}
#else
inline void ASSERT_CUDA(cudaError_t error_code) { (void)error_code; }
#endif
#endif
}

#endif
