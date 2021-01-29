///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

Exception::Exception(const std::string &msg, const char *file, const char *func, int line) {
  std::stringstream ss;
  ss << "In " << file << "\n";
  ss << func << " ";
  ss << line << "\n";
  ss << msg;
  msg_ = ss.str();
}

Exception::~Exception() NOEXCEPT {}

const char *Exception::what() const NOEXCEPT { return msg_.c_str(); }

}  // namespace crocoddyl