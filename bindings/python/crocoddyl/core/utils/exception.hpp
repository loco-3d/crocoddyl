///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2023-2023, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_EXCEPTION_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_EXCEPTION_HPP_

#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

static PyObject* createExceptionClass(const char* name,
                                      PyObject* base_type = PyExc_Exception) {
  const std::string scope_name =
      bp::extract<std::string>(bp::scope().attr("__name__"));
  const std::string qualified_name = scope_name + "." + name;
  PyObject* type = PyErr_NewException(qualified_name.c_str(), base_type, 0);
  if (!type) {
    bp::throw_error_already_set();
  }
  bp::scope().attr(name) = bp::handle<>(bp::borrowed(type));
  return type;
}

PyObject* ExceptionType = NULL;
void translateException(Exception const& e) {
  bp::object exc_t(bp::handle<>(bp::borrowed(ExceptionType)));
  exc_t.attr("cause") =
      bp::object(e);  // add the wrapped exception to the Python exception
  exc_t.attr("what") = bp::object(e.what());  // for convenience
  PyErr_SetString(
      ExceptionType,
      e.what());  // the string is used by print(exception) in python
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_EXCEPTION_HPP_
