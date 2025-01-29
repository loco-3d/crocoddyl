///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"

#include "crocoddyl/multibody/data/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorMultibody() {
  bp::class_<DataCollectorMultibody, bp::bases<DataCollectorAbstract> >(
      "DataCollectorMultibody", "Data collector for multibody systems.\n\n",
      bp::init<pinocchio::Data*>(
          bp::args("self", "pinocchio"),
          "Create multibody data collection.\n\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2>()])
      .add_property("pinocchio",
                    bp::make_getter(&DataCollectorMultibody::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .def(CopyableVisitor<DataCollectorMultibody>());

  bp::class_<DataCollectorActMultibody,
             bp::bases<DataCollectorMultibody, DataCollectorActuation> >(
      "DataCollectorActMultibody",
      "Data collector for actuated multibody systems.\n\n",
      bp::init<pinocchio::Data*, std::shared_ptr<ActuationDataAbstract> >(
          bp::args("self", "pinocchio", "actuation"),
          "Create multibody data collection.\n\n"
          ":param pinocchio: Pinocchio data\n"
          ":param actuation: actuation data")[bp::with_custodian_and_ward<1,
                                                                          2>()])
      .def(CopyableVisitor<DataCollectorActMultibody>());

  bp::class_<DataCollectorJointActMultibody,
             bp::bases<DataCollectorActMultibody, DataCollectorJoint> >(
      "DataCollectorJointActMultibody",
      "Data collector for actuated-joint multibody systems.\n\n",
      bp::init<pinocchio::Data*, std::shared_ptr<ActuationDataAbstract>,
               std::shared_ptr<JointDataAbstract> >(
          bp::args("self", "pinocchio", "actuation", "joint"),
          "Create multibody data collection.\n\n"
          ":param pinocchio: Pinocchio data\n"
          ":param actuation: actuation data\n"
          ":param joint: joint data")[bp::with_custodian_and_ward<1, 2>()])
      .def(CopyableVisitor<DataCollectorJointActMultibody>());
}

}  // namespace python
}  // namespace crocoddyl
