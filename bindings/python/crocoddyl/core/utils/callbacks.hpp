///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_UTILS_CALLBACKS_HPP_

#include "crocoddyl/core/utils/callbacks.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCallbacks() {
  bp::register_ptr_to_python<boost::shared_ptr<CallbackAbstract> >();

  bp::enum_<VerboseLevel>("VerboseLevel").value("_1", _1).value("_2", _2);

  bp::class_<CallbackVerbose, bp::bases<CallbackAbstract> >(
      "CallbackVerbose", "Callback function for printing the solver values.",
      bp::init<bp::optional<VerboseLevel> >(bp::args("self", "level"),
                                            "Initialize the differential verbose callback.\n\n"
                                            ":param level: verbose level (default _1)"))
      .def("__call__", &CallbackVerbose::operator(), bp::args("self", "solver"),
           "Run the callback function given a solver.\n\n"
           ":param solver: solver to be diagnostic");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
