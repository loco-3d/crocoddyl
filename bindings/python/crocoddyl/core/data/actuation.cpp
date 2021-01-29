///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/data/actuation.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorActuation() {
  bp::class_<DataCollectorActuation, bp::bases<DataCollectorAbstract> >(
      "DataCollectorActuation", "Actuation data collector.\n\n",
      bp::init<boost::shared_ptr<ActuationDataAbstract> >(bp::args("self", "actuation"),
                                                          "Create actuation data collection.\n\n"
                                                          ":param actuation: actuation data"))
      .add_property(
          "actuation",
          bp::make_getter(&DataCollectorActuation::actuation, bp::return_value_policy<bp::return_by_value>()),
          "actuation data");
}

}  // namespace python
}  // namespace crocoddyl
