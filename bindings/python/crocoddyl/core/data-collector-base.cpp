///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/data-collector-base.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollector() {
  bp::class_<DataCollectorAbstract>(
      "DataCollectorAbstract",
      "Abstract class for common collection of data used in different objects "
      "in action model.\n\n",
      bp::init<>(bp::args("self")))
      .def(CopyableVisitor<DataCollectorAbstract>());
}

}  // namespace python
}  // namespace crocoddyl
