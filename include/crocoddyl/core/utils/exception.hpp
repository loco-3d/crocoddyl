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
#include <iostream>
#include <sstream>

namespace crocoddyl {

class CrocoddylException : public std::exception {
 public:
  CrocoddylException(std::string message);
  ~CrocoddylException() throw();

  const char *what() const throw();
  std::string getMessage();

 private:
  std::string message;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_EXCEPTION_HPP_
