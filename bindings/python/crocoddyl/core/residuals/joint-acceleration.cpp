///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2023, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/residuals/joint-acceleration.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualJointAcceleration() {
  bp::register_ptr_to_python<std::shared_ptr<ResidualModelJointAcceleration>>();

  bp::class_<ResidualModelJointAcceleration, bp::bases<ResidualModelAbstract>>(
      "ResidualModelJointAcceleration",
      "This residual function defines a residual vector as r = a - aref, with "
      "a and aref as the current and\n"
      "reference joint acceleration (i.e., generalized acceleration), "
      "respectively.",
      bp::init<std::shared_ptr<StateAbstract>, Eigen::VectorXd, std::size_t>(
          bp::args("self", "state", "aref", "nu"),
          "Initialize the joint-acceleration residual model.\n\n"
          ":param state: state description\n"
          ":param aref: reference joint acceleration\n"
          ":param nu: dimension of the control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "aref"),
          "Initialize the joint-acceleration residual model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param aref: reference joint acceleration"))
      .def(bp::init<std::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the joint-acceleration residual model.\n\n"
          "The default reference joint-acceleration is obtained from "
          "np.zero(actuation.nu).\n"
          ":param state: state description\n"
          ":param nu: dimension of the control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract>>(
          bp::args("self", "state"),
          "Initialize the joint-acceleration residual model.\n\n"
          "The default reference joint-acceleration is obtained from "
          "np.zero(actuation.nu).\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description"))
      .def<void (ResidualModelJointAcceleration::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelJointAcceleration::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the joint-acceleration residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelJointAcceleration::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelJointAcceleration::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelJointAcceleration::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the joint-acceleration residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelJointAcceleration::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelJointAcceleration::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the joint-acceleration residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for the joint-acceleration residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelJointAcceleration::get_reference,
                            bp::return_internal_reference<>()),
          &ResidualModelJointAcceleration::set_reference,
          "reference joint acceleration")
      .def(CopyableVisitor<ResidualModelJointAcceleration>());
}

}  // namespace python
}  // namespace crocoddyl
