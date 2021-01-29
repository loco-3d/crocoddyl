///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#if __cplusplus < 201103L
#ifndef CROCODDYL_CORE_UTILS_TO_STRING_HPP_
#define CROCODDYL_CORE_UTILS_TO_STRING_HPP_

#include <sstream>

namespace std {

template <typename T> std::string to_string(T number) {
  std::ostringstream ss;
  ss << number;
  return ss.str();
}

} // namespace std

#endif // CROCODDYL_CORE_UTILS_TO_STRING_HPP_
#endif // __cplusplus__ >= 201103L
