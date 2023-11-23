///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_EXCEPTION_HPP_
#define CROCODDYL_CORE_UTILS_EXCEPTION_HPP_

#include <exception>
#include <iostream>
#include <sstream>

#define NOEXCEPT noexcept

#define throw_pretty(m)                                                 \
  {                                                                     \
    std::stringstream ss;                                               \
    ss << m;                                                            \
    throw crocoddyl::Exception(ss.str(), __FILE__, __PRETTY_FUNCTION__, \
                               __LINE__);                               \
  }

#ifndef NDEBUG
#define assert_pretty(condition, m)                                     \
  if (!(condition)) {                                                   \
    std::stringstream ss;                                               \
    ss << m;                                                            \
    throw crocoddyl::Exception(ss.str(), __FILE__, __PRETTY_FUNCTION__, \
                               __LINE__);                               \
  }
#else
#define assert_pretty(condition, m) ((void)0)
#endif
namespace crocoddyl {

class Exception : public std::exception {
 public:
  explicit Exception(const std::string &msg, const char *file, const char *func,
                     int line);
  virtual ~Exception() NOEXCEPT;
  virtual const char *what() const NOEXCEPT;

  std::string getMessage() const;
  std::string getExtraData() const;

 private:
  std::string exception_msg_;
  std::string extra_data_;
  std::string msg_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_EXCEPTION_HPP_
