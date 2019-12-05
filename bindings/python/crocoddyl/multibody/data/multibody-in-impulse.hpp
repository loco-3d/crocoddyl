///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_IN_IMPULSE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_IN_IMPULSE_HPP_

#include "crocoddyl/multibody/data/multibody-in-impulse.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorMultibodyInImpulse() {
  bp::class_<DataCollectorMultibodyInImpulse, bp::bases<DataCollectorMultibody> >(
      "DataCollectorMultibodyInImpulse", "Class for common multibody in impulse data between cost functions.\n\n",
      bp::init<pinocchio::Data*, boost::shared_ptr<ImpulseDataMultiple> >(
          bp::args("self", "data", "impulses"),
          "Create multibody shared data.\n\n"
          ":param data: Pinocchio data\n"
          ":param impulses: impulses data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "impulses",
          bp::make_getter(&DataCollectorMultibodyInImpulse::impulses, bp::return_value_policy<bp::return_by_value>()),
          "impulses data");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_IN_IMPULSE_HPP_
