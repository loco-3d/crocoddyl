///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

Exception::Exception(const std::string &msg, const char *file, const char *func,
                     int line) {
  std::stringstream ss;
  ss << "In " << file << "\n";
  ss << func << " ";
  ss << line << "\n";
  ss << msg;
  msg_ = ss.str();
  exception_msg_ = msg;
  extra_data_ = file;
}

Exception::~Exception() NOEXCEPT {}

const char *Exception::what() const NOEXCEPT { return msg_.c_str(); }

std::string Exception::getMessage() const { return exception_msg_; }

std::string Exception::getExtraData() const { return extra_data_; }

}  // namespace crocoddyl
