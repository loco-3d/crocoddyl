///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_TO_STRING_HPP_
#define CROCODDYL_CORE_UTILS_TO_STRING_HPP_

#include <sstream>

namespace crocoddyl {

template <typename T>
std::string to_string(T Number) {
  std::ostringstream ss;
  ss << Number;
  return ss.str();
}

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_TO_STRING_HPP_
