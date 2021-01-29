///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, INRIA, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_DEPRECATE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_DEPRECATE_HPP_

#include <Python.h>
#include <boost/python.hpp>
#include <string>

namespace crocoddyl {
namespace python {
template <class Policy = boost::python::default_call_policies>
struct deprecated : Policy {
  deprecated(const std::string &warning_message = "")
      : Policy(), m_warning_message(warning_message) {}

  template <class ArgumentPackage>
  bool precall(ArgumentPackage const &args) const {
    PyErr_WarnEx(PyExc_UserWarning, m_warning_message.c_str(), 1);
    return static_cast<const Policy *>(this)->precall(args);
  }

  typedef typename Policy::result_converter result_converter;
  typedef typename Policy::argument_package argument_package;

protected:
  const std::string m_warning_message;
};
} // namespace python
} // namespace crocoddyl

#endif // BINDINGS_PYTHON_CROCODDYL_UTILS_DEPRECATE_HPP_