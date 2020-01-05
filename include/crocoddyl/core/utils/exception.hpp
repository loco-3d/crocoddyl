///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_EXCEPTION_HPP_
#define CROCODDYL_CORE_UTILS_EXCEPTION_HPP_

#include <exception>
#include <sstream>
#include "crocoddyl/core/utils/to-string.hpp"

#if __cplusplus >= 201103L  // We are using C++11 or a later version
#define NOEXCEPT noexcept
#else
#define NOEXCEPT throw()
#endif

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
  virtual ~Exception() NOEXCEPT;
  virtual const char *what() const NOEXCEPT;

  std::string msg_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_EXCEPTION_HPP_