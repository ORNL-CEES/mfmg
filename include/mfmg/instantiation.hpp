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

#ifndef AMGE_INSTANTIATION_HPP
#define AMGE_INSTANTIATION_HPP

#include <boost/preprocessor/repeat_from_to.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/replace.hpp>

// TUPLE(class_name) expend to (class_name, 0)
#define TUPLE_N (0, 0)
#define TUPLE(class_name) BOOST_PP_TUPLE_REPLACE(TUPLE_N, 0, class_name)
#define NUMBER_TYPE (float)(double)

// Instantiation of the class for every dim
#define M_DIM_INSTANT(z, dim, CLASS_NAME_TUPLE)                                \
  template class mfmg::BOOST_PP_TUPLE_ELEM(0, CLASS_NAME_TUPLE)<dim>;
// CLASS_NAME_TUPLE (class_name, 0)
#define INSTANTIATE_DIM(CLASS_NAME_TUPLE)                                      \
  BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM_INSTANT, CLASS_NAME_TUPLE)

// Instantiation of the class for every dim-NumberType combination
#define M_DIM_NUMBERTYPE_INSTANT(z, CLASS_NAME_DIM_TUPLE, number_type)         \
  template class mfmg::BOOST_PP_TUPLE_ELEM(                                    \
      0, CLASS_NAME_DIM_TUPLE)<BOOST_PP_TUPLE_ELEM(1, CLASS_NAME_DIM_TUPLE),   \
                               number_type>;
// CLASS_NAME_DIM_TUPLE (class_name, 2) (class_name, 3)
#define M_NUMBERTYPE(z, dim, CLASS_NAME_TUPLE)                                 \
  BOOST_PP_SEQ_FOR_EACH(M_DIM_NUMBERTYPE_INSTANT,                              \
                        BOOST_PP_TUPLE_REPLACE(CLASS_NAME_TUPLE, 1, dim),      \
                        NUMBER_TYPE)
// CLASS_NAME_TUPLE (class_name, 0)
#define INSTANTIATE_DIM_NUMBERTYPE(CLASS_NAME_TUPLE)                           \
  BOOST_PP_REPEAT_FROM_TO(2, 4, M_NUMBERTYPE, CLASS_NAME_TUPLE)

#endif
