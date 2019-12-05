///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_IN_CONTACT_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_IN_CONTACT_HPP_

#include "crocoddyl/multibody/data/multibody-in-contact.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorMultibodyInContact() {
  bp::class_<DataCollectorMultibodyInContact, bp::bases<DataCollectorMultibody> >(
      "DataCollectorMultibodyInContact", "Class for common multibody in contact data between cost functions.\n\n",
      bp::init<pinocchio::Data*, boost::shared_ptr<ContactDataMultiple> >(
          bp::args("self", "data", "contacts"),
          "Create multibody shared data.\n\n"
          ":param data: Pinocchio data\n"
          ":param contacts: contacts data")[bp::with_custodian_and_ward<1, 2>()])
      .add_property(
          "contacts",
          bp::make_getter(&DataCollectorMultibodyInContact::contacts, bp::return_value_policy<bp::return_by_value>()),
          "contacts data");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_MULTIBODY_IN_CONTACT_HPP_
