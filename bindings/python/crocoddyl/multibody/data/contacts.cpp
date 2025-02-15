///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/data/contacts.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorContacts() {
  bp::class_<DataCollectorContact, bp::bases<DataCollectorAbstract> >(
      "DataCollectorContact", "Contact data collector.\n\n",
      bp::init<std::shared_ptr<ContactDataMultiple> >(
          bp::args("self", "contacts"),
          "Create contact data collection.\n\n"
          ":param contacts: contacts data"))
      .add_property(
          "contacts",
          bp::make_getter(&DataCollectorContact::contacts,
                          bp::return_value_policy<bp::return_by_value>()),
          "contacts data")
      .def(CopyableVisitor<DataCollectorContact>());

  bp::class_<DataCollectorMultibodyInContact,
             bp::bases<DataCollectorMultibody, DataCollectorContact> >(
      "DataCollectorMultibodyInContact",
      "Data collector for multibody systems in contact.\n\n",
      bp::init<pinocchio::Data*, std::shared_ptr<ContactDataMultiple> >(
          bp::args("self", "pinocchio", "contacts"),
          "Create multibody data collection.\n\n"
          ":param pinocchio: Pinocchio data\n"
          ":param contacts: contacts data")[bp::with_custodian_and_ward<1,
                                                                        2>()])
      .def(CopyableVisitor<DataCollectorMultibodyInContact>());

  bp::class_<
      DataCollectorActMultibodyInContact,
      bp::bases<DataCollectorMultibodyInContact, DataCollectorActuation> >(
      "DataCollectorActMultibodyInContact",
      "Data collector for actuated multibody systems in contact.\n\n",
      bp::init<pinocchio::Data*, std::shared_ptr<ActuationDataAbstract>,
               std::shared_ptr<ContactDataMultiple> >(
          bp::args("self", "pinocchio", "actuation", "contacts"),
          "Create multibody data collection.\n\n"
          ":param pinocchio: Pinocchio data\n"
          ":param actuation: actuation data\n"
          ":param contacts: contacts data")[bp::with_custodian_and_ward<1,
                                                                        2>()])
      .def(CopyableVisitor<DataCollectorActMultibodyInContact>());

  bp::class_<
      DataCollectorJointActMultibodyInContact,
      bp::bases<DataCollectorActMultibodyInContact, DataCollectorJoint> >(
      "DataCollectorJointActMultibodyInContact",
      "Data collector for actuated-joint multibody systems in contact.\n\n",
      bp::init<pinocchio::Data*, std::shared_ptr<ActuationDataAbstract>,
               std::shared_ptr<JointDataAbstract>,
               std::shared_ptr<ContactDataMultiple> >(
          bp::args("self", "pinocchio", "actuation", "joint", "contacts"),
          "Create multibody data collection.\n\n"
          ":param pinocchio: Pinocchio data\n"
          ":param actuation: actuation data\n"
          ":param joint: joint data\n"
          ":param contacts: contacts data")[bp::with_custodian_and_ward<1,
                                                                        2>()])
      .def(CopyableVisitor<DataCollectorJointActMultibodyInContact>());
}

}  // namespace python
}  // namespace crocoddyl
