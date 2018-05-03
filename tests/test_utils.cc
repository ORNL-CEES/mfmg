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
