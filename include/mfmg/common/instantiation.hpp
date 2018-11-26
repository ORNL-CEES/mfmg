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

#ifndef AMGE_INSTANTIATION_HPP
#define AMGE_INSTANTIATION_HPP

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_solver.h>

#include <boost/preprocessor.hpp>

#define TUPLE_0 (0, 0, 0, 0)
#define TUPLE_1 (0, 0, 0, 1)
// TUPLE(class_name) expand to (class_name, 0, 0, 0)
#define TUPLE(class_name) BOOST_PP_TUPLE_REPLACE(TUPLE_0, 0, class_name)
// TUPLE_PARAM(class_name, template_param) expand to (class_name, 0,
// template_param, 1)
#define TUPLE_PARAM(class_name, template_param)                                \
  BOOST_PP_TUPLE_REPLACE(BOOST_PP_TUPLE_REPLACE(TUPLE_1, 2, template_param),   \
                         0, class_name)

#define SCALAR_TYPE (float)(double)
#define VECTOR_TYPE (dealii::LinearAlgebra::distributed::Vector<double>)
#define SERIAL_VECTOR_TYPE (dealii::Vector<double>)

//////////////////////////////////////////////
// Instantiation of the class for every dim //
//////////////////////////////////////////////

#define M_DIM_INSTANT(z, dim, CLASS_NAME_TUPLE)                                \
  template class mfmg::BOOST_PP_TUPLE_ELEM(0, CLASS_NAME_TUPLE)<dim>;
// CLASS_NAME_TUPLE (class_name, 0)
#define INSTANTIATE_DIM(CLASS_NAME_TUPLE)                                      \
  BOOST_PP_REPEAT_FROM_TO(2, 4, M_DIM_INSTANT, CLASS_NAME_TUPLE)

/////////////////////////////////////////////////////
// Instantiation of the class for every VectorType //
/////////////////////////////////////////////////////

#define M_VECTORTYPE_INSTANT(z, CLASS_NAME_TUPLE, vector_type)                 \
  template class mfmg::BOOST_PP_TUPLE_ELEM(0, CLASS_NAME_TUPLE)<vector_type>;
// CLASS_NAME_TUPLE (class_name, 0)
#define INSTANTIATE_VECTORTYPE(CLASS_NAME_TUPLE)                               \
  BOOST_PP_SEQ_FOR_EACH(M_VECTORTYPE_INSTANT, CLASS_NAME_TUPLE, VECTOR_TYPE)

///////////////////////////////////////////////////////////
// Instantiation of the class for every SerialVectorType //
///////////////////////////////////////////////////////////

#define M_SERIALVECTORTYPE_INSTANT(z, CLASS_NAME_TUPLE, vector_type)           \
  template class mfmg::BOOST_PP_TUPLE_ELEM(0, CLASS_NAME_TUPLE)<vector_type>;
// CLASS_NAME_TUPLE (class_name, 0)
#define INSTANTIATE_SERIALVECTORTYPE(CLASS_NAME_TUPLE)                         \
  BOOST_PP_SEQ_FOR_EACH(M_SERIALVECTORTYPE_INSTANT, CLASS_NAME_TUPLE,          \
                        SERIAL_VECTOR_TYPE)

/////////////////////////////////////////////////////////////////////
// Instantiation of the class for every dim-ScalarType combination //
/////////////////////////////////////////////////////////////////////

#define M_DIM_SCALARTYPE_INSTANT(z, CLASS_NAME_DIM_TUPLE, scalar_type)         \
  template class mfmg::BOOST_PP_TUPLE_ELEM(                                    \
      0, CLASS_NAME_DIM_TUPLE)<BOOST_PP_TUPLE_ELEM(1, CLASS_NAME_DIM_TUPLE),   \
                               scalar_type>;
// CLASS_NAME_DIM_TUPLE (class_name, 2) (class_name, 3)
#define M_SCALARTYPE(z, dim, CLASS_NAME_TUPLE)                                 \
  BOOST_PP_SEQ_FOR_EACH(M_DIM_SCALARTYPE_INSTANT,                              \
                        BOOST_PP_TUPLE_REPLACE(CLASS_NAME_TUPLE, 1, dim),      \
                        SCALAR_TYPE)
// CLASS_NAME_TUPLE (class_name, 0)
#define INSTANTIATE_DIM_SCALARTYPE(CLASS_NAME_TUPLE)                           \
  BOOST_PP_REPEAT_FROM_TO(2, 4, M_SCALARTYPE, CLASS_NAME_TUPLE)

/////////////////////////////////////////////////////////////////////
// Instantiation of the class for every dim-VectorType combination //
/////////////////////////////////////////////////////////////////////

// Because BOOST_PP_COMMA expands to comma only when followed by () and that we
// cannot expand the macro inside BOOST_PP_IF we need to split the instantiation
// in two different macros
#define M_PARAM_INSTANT_1(CLASS_NAME_DIM_TUPLE)                                \
  mfmg::BOOST_PP_TUPLE_ELEM(2, CLASS_NAME_DIM_TUPLE) <                         \
      BOOST_PP_TUPLE_ELEM(1, CLASS_NAME_DIM_TUPLE) BOOST_PP_COMMA

#define M_PARAM_INSTANT_2(vector_type) vector_type > BOOST_PP_COMMA

// Instantiation of the class for every dim-VectorType combination
#define M_DIM_VECTORTYPE_INSTANT(z, CLASS_NAME_DIM_TUPLE, vector_type)         \
  template class mfmg::BOOST_PP_TUPLE_ELEM(0, CLASS_NAME_DIM_TUPLE)<           \
      BOOST_PP_TUPLE_ELEM(1, CLASS_NAME_DIM_TUPLE),                            \
      BOOST_PP_IF(BOOST_PP_TUPLE_ELEM(3, CLASS_NAME_DIM_TUPLE),                \
                  M_PARAM_INSTANT_1(CLASS_NAME_DIM_TUPLE), BOOST_PP_EMPTY)()   \
          BOOST_PP_IF(BOOST_PP_TUPLE_ELEM(3, CLASS_NAME_DIM_TUPLE),            \
                      M_PARAM_INSTANT_2(vector_type), BOOST_PP_EMPTY)()        \
              vector_type>;

// CLASS_NAME_DIM_TUPLE (class_name, 2) (class_name, 3)
#define M_VECTORTYPE(z, dim, CLASS_NAME_TUPLE)                                 \
  BOOST_PP_SEQ_FOR_EACH(M_DIM_VECTORTYPE_INSTANT,                              \
                        BOOST_PP_TUPLE_REPLACE(CLASS_NAME_TUPLE, 1, dim),      \
                        VECTOR_TYPE)

// CLASS_NAME_TUPLE (class_name, 0)
#define INSTANTIATE_DIM_VECTORTYPE(CLASS_NAME_TUPLE)                           \
  BOOST_PP_REPEAT_FROM_TO(2, 4, M_VECTORTYPE, CLASS_NAME_TUPLE)

#endif
