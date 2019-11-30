///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_UTILS_EXCEPTION_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_UTILS_EXCEPTION_HPP_

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

PyObject *CrocoddylExceptionType = NULL;

void translateCrocoddylException(CrocoddylException const &e) {
  assert(CrocoddylExceptionType != NULL);
  boost::python::object pythonExceptionInstance(e);
  PyErr_SetObject(CrocoddylExceptionType, pythonExceptionInstance.ptr());
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_UTILS_EXCEPTION_HPP_
