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
      "DataCollectorMultibody", "Class for common multibody data between cost functions.\n\n",
      bp::init<pinocchio::Data*>(bp::args("self", "data"),
                                 "Create multibody shared data.\n\n"
                                 ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2>()])
      .add_property("pinocchio",
                    bp::make_getter(&DataCollectorMultibody::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_HPP_
