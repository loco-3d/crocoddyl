///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/data/joint.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollectorJoint() {
  bp::register_ptr_to_python<boost::shared_ptr<JointDataAbstract> >();

  bp::class_<JointDataAbstract, boost::noncopyable>(
      "JointDataAbstract",
      "Abstract class for joint datas.\n\n"
      "A joint data contains all the required information about joint torques and accelerations.\n"
      "The joint data typically is allocated onces by running model.createData().",
      bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActuationModelAbstract>, std::size_t>(
          bp::args("self", "state", "actuation", "nu"),
          "Create the joint data.\n\n"
          "The joint data uses the model in order to first process it.\n"
          ":param state: state description\n"
          ":param actuation: actuation model\n"
          ":param nu: dimension of control vector."))
      .add_property("tau", bp::make_getter(&JointDataAbstract::tau, bp::return_internal_reference<>()),
                    bp::make_setter(&JointDataAbstract::tau), "joint torques")
      .add_property("a", bp::make_getter(&JointDataAbstract::a, bp::return_internal_reference<>()),
                    bp::make_setter(&JointDataAbstract::a), "generalized joint accelerations")
      .add_property("dtau_dx", bp::make_getter(&JointDataAbstract::dtau_dx, bp::return_internal_reference<>()),
                    bp::make_setter(&JointDataAbstract::dtau_dx),
                    "partial derivatives of the joint torques w.r.t. the state point")
      .add_property("dtau_du", bp::make_getter(&JointDataAbstract::dtau_du, bp::return_internal_reference<>()),
                    bp::make_setter(&JointDataAbstract::dtau_du),
                    "partial derivatives of the joint torques w.r.t. the control input")
      .add_property("da_dx", bp::make_getter(&JointDataAbstract::da_dx, bp::return_internal_reference<>()),
                    bp::make_setter(&JointDataAbstract::da_dx),
                    "partial derivatives of the generalized joint accelerations w.r.t. the state point")
      .add_property("da_du", bp::make_getter(&JointDataAbstract::da_du, bp::return_internal_reference<>()),
                    bp::make_setter(&JointDataAbstract::da_du),
                    "partial derivatives of the generalized joint accelerations w.r.t. the control input");

  bp::class_<DataCollectorJoint, bp::bases<DataCollectorAbstract> >(
      "DataCollectorJoint", "Joint data collector.\n\n",
      bp::init<boost::shared_ptr<JointDataAbstract> >(bp::args("self", "joint"),
                                                      "Create joint data collection.\n\n"
                                                      ":param joint: joint data"))
      .add_property("joint",
                    bp::make_getter(&DataCollectorJoint::joint, bp::return_value_policy<bp::return_by_value>()),
                    "joint data");

  bp::class_<DataCollectorJointActuation, bp::bases<DataCollectorActuation> >(
      "DataCollectorJointActuation", "Joint-actuation data collector.\n\n",
      bp::init<boost::shared_ptr<ActuationDataAbstract>, boost::shared_ptr<JointDataAbstract> >(
          bp::args("self", "actuation", "joint"),
          "Create joint-actuation data collection.\n\n"
          ":param actuation: actuation data"
          ":param joint: joint data"));
}

}  // namespace python
}  // namespace crocoddyl
