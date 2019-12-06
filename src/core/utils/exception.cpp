///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

Exception::Exception(const std::string &msg, const char *file, const char *func, int line) {
  msg_ = "In " + std::string(file) + "\n";
  msg_ += std::string(func) + " ";
  msg_ += std::to_string(line) + "\n";
  msg_ += msg;
}

Exception::~Exception() NOEXCEPT {}

const char *Exception::what() const NOEXCEPT { return msg_.c_str(); }

}  // namespace crocoddyl