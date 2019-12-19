///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_DATA_COLLECTOR_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_DATA_COLLECTOR_BASE_HPP_

#include "crocoddyl/core/data-collector-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeDataCollector() {
  bp::class_<DataCollectorAbstract, boost::noncopyable>(
      "DataCollectorAbstract",
      "Abstract class for common collection of data used in different objects in action model.\n\n",
      bp::init<>(bp::args("self")));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_DATA_COLLECTOR_BASE_HPP_
