///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/residuals/joint-torque.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualJointTorque() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelJointTorque> >();

  bp::class_<ResidualModelJointTorque, bp::bases<ResidualModelAbstract> >(
      "ResidualModelJointTorque",
      "This residual function defines a residual vector as r = u - uref, with u and uref as the current and\n"
      "reference joint torques, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActuationModelAbstract>, Eigen::VectorXd,
               std::size_t>(bp::args("self", "state", "actuation", "uref", "nu"),
                            "Initialize the joint-torque residual model.\n\n"
                            ":param state: state description\n"
                            ":param actuation: actuation model\n"
                            ":param uref: reference joint torque\n"
                            ":param nu: dimension of the control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActuationModelAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "actuation", "uref"),
          "Initialize the joint-torque residual model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param actuation: actuation model\n"
          ":param uref: reference joint torque"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActuationModelAbstract>, std::size_t>(
          bp::args("self", "state", "actuation", "nu"),
          "Initialize the joint-torque residual model.\n\n"
          "The default reference joint-torque is obtained from np.zero(actuation.nu).\n"
          ":param state: state description\n"
          ":param actuation: actuation model\n"
          ":param nu: dimension of the control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActuationModelAbstract> >(
          bp::args("self", "state", "actuation"),
          "Initialize the joint-torque residual model.\n\n"
          "The default reference joint-torque is obtained from np.zero(actuation.nu).\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param actuation: actuation model"))
      .def<void (ResidualModelJointTorque::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelJointTorque::calc, bp::args("self", "data", "x", "u"),
          "Compute the joint-torque residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelJointTorque::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelJointTorque::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelJointTorque::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the joint-torque residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelJointTorque::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelJointTorque::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the joint-torque residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the joint-torque residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("reference",
                    bp::make_function(&ResidualModelJointTorque::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelJointTorque::set_reference, "reference joint torque");
}

}  // namespace python
}  // namespace crocoddyl
