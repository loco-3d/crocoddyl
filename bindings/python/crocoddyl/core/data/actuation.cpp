///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/data/actuation.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorActuation() {
  bp::class_<DataCollectorActuation, bp::bases<DataCollectorAbstract> >(
      "DataCollectorActuation", "Actuation data collector.\n\n",
      bp::init<std::shared_ptr<ActuationDataAbstract> >(
          bp::args("self", "actuation"),
          "Create actuation data collection.\n\n"
          ":param actuation: actuation data"))
      .add_property(
          "actuation",
          bp::make_getter(&DataCollectorActuation::actuation,
                          bp::return_value_policy<bp::return_by_value>()),
          "actuation data")
      .def(CopyableVisitor<DataCollectorActuation>());
}

}  // namespace python
}  // namespace crocoddyl
