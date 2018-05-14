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

#define BOOST_TEST_MODULE utils

#include "main.cc"

#include <mfmg/utils.hpp>

#include <Teuchos_ParameterList.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

BOOST_AUTO_TEST_CASE(plist2ptree)
{
  boost::property_tree::ptree ptree;
  Teuchos::ParameterList plist;

  std::string info_string = R"(
eigensolver
{
  "number of eigenvectors" 2
  tolerance 1e-14
}
smoother
{
  type "Gauss-Seidel"
}
"is preconditioner" false
{
}
coarse
{
  type "ml"
  params
  {
    verbosity 5
    "smoother: type" "Gauss-Seidel"
  }
})";

  std::istringstream is(info_string);
  boost::property_tree::info_parser::read_info(is, ptree);

  mfmg::ptree2plist(ptree, plist);

  BOOST_TEST(plist.sublist("eigensolver").get<int>("number of eigenvectors") ==
             2);
  BOOST_TEST(plist.sublist("eigensolver").get<double>("tolerance") == 1e-14);
  BOOST_TEST(plist.get<bool>("is preconditioner") == false);
  BOOST_TEST(plist.sublist("coarse").sublist("params").get<int>("verbosity") ==
             5);
}

BOOST_AUTO_TEST_CASE(sorting)
{
  unsigned int const size = 10;
  std::vector<int> vec_1(size);
  std::vector<int> vec_2(size);

  for (unsigned int i = 0; i < size; ++i)
  {
    vec_1[i] = i % 2;
    vec_2[i] = i;
  }

  auto permutation = mfmg::sort_permutation(vec_1, vec_2);

  mfmg::apply_permutation_in_place(permutation, vec_2);
  std::vector<int> reference(size);
  for (unsigned int i = 0; i < size / 2; ++i)
  {
    reference[i] = 2 * i;
    reference[i + size / 2] = 2 * i + 1;
  }

  BOOST_TEST(reference == vec_2);
}
