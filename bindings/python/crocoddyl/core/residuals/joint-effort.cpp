///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2023, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/residuals/joint-effort.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualJointEffort() {
  bp::register_ptr_to_python<std::shared_ptr<ResidualModelJointEffort> >();

  bp::class_<ResidualModelJointEffort, bp::bases<ResidualModelAbstract> >(
      "ResidualModelJointEffort",
      "This residual function defines a residual vector as r = u - uref, with "
      "u and uref as the current and\n"
      "reference joint efforts, respectively.",
      bp::init<std::shared_ptr<StateAbstract>,
               std::shared_ptr<ActuationModelAbstract>, Eigen::VectorXd,
               std::size_t, bp::optional<bool> >(
          bp::args("self", "state", "actuation", "uref", "nu", "fwddyn"),
          "Initialize the joint-effort residual model.\n\n"
          ":param state: state description\n"
          ":param actuation: actuation model\n"
          ":param uref: reference joint effort\n"
          ":param nu: dimension of the control vector\n"
          ":param fwddyn: indicate if we have a forward dynamics problem "
          "(True) or inverse "
          "dynamics problem (False) (default False)"))
      .def(bp::init<std::shared_ptr<StateAbstract>,
                    std::shared_ptr<ActuationModelAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "actuation", "uref"),
          "Initialize the joint-effort residual model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param actuation: actuation model\n"
          ":param uref: reference joint effort"))
      .def(bp::init<std::shared_ptr<StateAbstract>,
                    std::shared_ptr<ActuationModelAbstract>, std::size_t>(
          bp::args("self", "state", "actuation", "nu"),
          "Initialize the joint-effort residual model.\n\n"
          "The default reference joint-effort is obtained from "
          "np.zero(actuation.nu).\n"
          ":param state: state description\n"
          ":param actuation: actuation model\n"
          ":param nu: dimension of the control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract>,
                    std::shared_ptr<ActuationModelAbstract> >(
          bp::args("self", "state", "actuation"),
          "Initialize the joint-effort residual model.\n\n"
          "The default reference joint-effort is obtained from "
          "np.zero(actuation.nu).\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param actuation: actuation model"))
      .def<void (ResidualModelJointEffort::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelJointEffort::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the joint-effort residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelJointEffort::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelJointEffort::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelJointEffort::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the joint-effort residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelJointEffort::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelJointEffort::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the joint-effort residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for the joint-effort residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("reference",
                    bp::make_function(&ResidualModelJointEffort::get_reference,
                                      bp::return_internal_reference<>()),
                    &ResidualModelJointEffort::set_reference,
                    "reference joint effort")
      .def(CopyableVisitor<ResidualModelJointEffort>());
}

}  // namespace python
}  // namespace crocoddyl
