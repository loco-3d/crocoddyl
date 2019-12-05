///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_CONTACTS_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_CONTACTS_HPP_

#include "crocoddyl/multibody/data/contacts.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorContacts() {
  bp::class_<DataCollectorContact, bp::bases<DataCollectorAbstract> >(
      "DataCollectorContact", "Class for common contact data between cost functions.\n\n",
      bp::init<boost::shared_ptr<ContactDataMultiple> >(bp::args("self", "contacts"),
                                                        "Create contact shared data.\n\n"
                                                        ":param contacts: contacts data"))
      .add_property("contacts",
                    bp::make_getter(&DataCollectorContact::contacts, bp::return_value_policy<bp::return_by_value>()),
                    "contacts data");

  bp::class_<DataCollectorMultibodyInContact, bp::bases<DataCollectorMultibody, DataCollectorContact> >(
      "DataCollectorMultibodyInContact", "Class for common multibody in contact data between cost functions.\n\n",
      bp::init<pinocchio::Data*, boost::shared_ptr<ContactDataMultiple> >(
          bp::args("self", "data", "contacts"),
          "Create multibody shared data.\n\n"
          ":param data: Pinocchio data\n"
          ":param contacts: contacts data")[bp::with_custodian_and_ward<1, 2>()]);
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_DATA_CONTACTS_HPP_
