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

#include <mfmg/common/utils.hpp>

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include <string>

namespace mfmg
{
static std::ostream &
ptree2plist_internal(boost::property_tree::ptree const &node, std::ostream &os)
{
  ASSERT(!node.empty(), "Internal error");

  // Iterate on children
  for (auto const &item : node)
  {
    bool child_is_sublist = !item.second.empty();
    if (child_is_sublist)
    {
      os << "<ParameterList name=\"" << item.first << "\">" << std::endl;
      ptree2plist_internal(item.second, os);
      os << "</ParameterList>" << std::endl;
    }
    else
    {
      auto value = item.second.data();
      std::string value_type;

      if (item.second.get_value_optional<int>())
      {
        value_type = "int";
      }
      else if (item.second.get_value_optional<double>())
      {
        value_type = "double";
      }
      else if (item.second.get_value_optional<bool>())
      {
        value_type = "bool";
      }
      else
      {
        value_type = "string";
      }

      os << "<Parameter name=\"" << item.first << "\" type=\"" << value_type
         << "\" value=\"" << value << "\"/>" << std::endl;
    }
  }
  return os;
}

void ptree2plist(boost::property_tree::ptree const &ptree,
                 Teuchos::ParameterList &plist)
{
  if (ptree.empty())
  {
    plist = Teuchos::ParameterList();
    return;
  }

  std::ostringstream ss;

  ss << "<ParameterList name=\"ANONYMOUS\">\n";
  ptree2plist_internal(ptree, ss);
  ss << "</ParameterList>";

  plist = *Teuchos::getParametersFromXmlString(ss.str());
}
} // namespace mfmg
