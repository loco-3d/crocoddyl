///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_IMPULSES_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_IMPULSES_HPP_

#include "crocoddyl/multibody/data/impulses.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorImpulses() {
  bp::class_<DataCollectorImpulse, bp::bases<DataCollectorAbstract> >(
      "DataCollectorImpulse", "Impulse data collector.\n\n",
      bp::init<boost::shared_ptr<ImpulseDataMultiple> >(bp::args("self", "impulses"),
                                                        "Create impulse data collection.\n\n"
                                                        ":param impulses: impulses data"))
      .add_property("impulses",
                    bp::make_getter(&DataCollectorImpulse::impulses, bp::return_value_policy<bp::return_by_value>()),
                    "impulses data");

  bp::class_<DataCollectorMultibodyInImpulse, bp::bases<DataCollectorMultibody, DataCollectorImpulse> >(
      "DataCollectorMultibodyInImpulse", "Data collector for multibody systems in impulse.\n\n",
      bp::init<pinocchio::Data*, boost::shared_ptr<ImpulseDataMultiple> >(
          bp::args("self", "pinocchio", "impulses"),
          "Create multibody data collection.\n\n"
          ":param pinocchio: Pinocchio data\n"
          ":param impulses: impulses data")[bp::with_custodian_and_ward<1, 2>()]);
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_IMPULSES_HPP_
