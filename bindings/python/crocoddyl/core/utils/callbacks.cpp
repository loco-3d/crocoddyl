///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/callbacks.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

void exposeCallbacks() {
  bp::register_ptr_to_python<boost::shared_ptr<CallbackAbstract>>();

  bp::enum_<VerboseLevel>("VerboseLevel").value("_1", _1).value("_2", _2);

  bp::class_<CallbackVerbose, bp::bases<CallbackAbstract>>(
      "CallbackVerbose", "Callback function for printing the solver values.",
      bp::init<bp::optional<VerboseLevel>>(
          bp::args("self", "level"),
          "Initialize the differential verbose callback.\n\n"
          ":param level: verbose level (default _1)"))
      .def("__call__", &CallbackVerbose::operator(), bp::args("self", "solver"),
           "Run the callback function given a solver.\n\n"
           ":param solver: solver to be diagnostic");
}

} // namespace python
} // namespace crocoddyl
