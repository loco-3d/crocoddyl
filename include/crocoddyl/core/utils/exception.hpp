///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_EXCEPTION_HPP_
#define CROCODDYL_CORE_UTILS_EXCEPTION_HPP_

#include <exception>
#include <sstream>

#define throw_pretty(m)                                                            \
  {                                                                                \
    std::stringstream ss;                                                          \
    ss << m;                                                                       \
    throw crocoddyl::Exception(ss.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__); \
  }

#define assert_pretty(condition, m)                                                \
  if (!(condition)) {                                                              \
    std::stringstream ss;                                                          \
    ss << m;                                                                       \
    throw crocoddyl::Exception(ss.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__); \
  }

namespace crocoddyl {

class Exception : public std::exception {
 public:
  explicit Exception(const std::string &msg, const char *file, const char *func, int line);
  virtual const char *what() const noexcept;

  std::string msg_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_EXCEPTION_HPP_