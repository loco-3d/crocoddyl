///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_HPP_

#include "crocoddyl/multibody/data/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorMultibody() {
  bp::class_<DataCollectorMultibody, bp::bases<DataCollectorAbstract> >(
      "DataCollectorMultibody", "Data collector for multibody systems.\n\n",
      bp::init<pinocchio::Data*>(bp::args("self", "pinocchio"),
                                 "Create multibody data collection.\n\n"
                                 ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2>()])
      .add_property("pinocchio",
                    bp::make_getter(&DataCollectorMultibody::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data");

  bp::class_<DataCollectorActMultibody, bp::bases<DataCollectorMultibody, DataCollectorActuation> >(
      "DataCollectorActMultibody", "Data collector for actuated multibody systems.\n\n",
      bp::init<pinocchio::Data*, boost::shared_ptr<ActuationDataAbstract> >(
          bp::args("self", "pinocchio", "actuation"),
          "Create multibody data collection.\n\n"
          ":param pinocchio: Pinocchio data\n"
          ":param actuation: actuation data")[bp::with_custodian_and_ward<1, 2>()]);
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_HPP_
