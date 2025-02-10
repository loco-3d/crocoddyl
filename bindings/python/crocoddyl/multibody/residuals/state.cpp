///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/state.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualState() {
  bp::register_ptr_to_python<std::shared_ptr<ResidualModelState> >();

  bp::class_<ResidualModelState, bp::bases<ResidualModelAbstract> >(
      "ResidualModelState",
      "This cost function defines a residual vector as r = x - xref, with x "
      "and xref as the current and reference\n"
      "state, respectively.",
      bp::init<std::shared_ptr<StateAbstract>, Eigen::VectorXd, std::size_t>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the state cost model.\n\n"
          ":param state: state description\n"
          ":param xref: reference state (default state.zero())\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "xref"),
          "Initialize the state cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param xref: reference state"))
      .def(bp::init<std::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero().\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero(), and nu "
          "from state.nv.\n"
          ":param state: state description"))
      .def<void (ResidualModelState::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelState::calc, bp::args("self", "data", "x", "u"),
          "Compute the state cost.\n\n"
          ":param data: cost data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelState::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelState::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelState::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the state cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelState::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelState::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the stae residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for the control residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("reference",
                    bp::make_function(&ResidualModelState::get_reference,
                                      bp::return_internal_reference<>()),
                    &ResidualModelState::set_reference, "reference state")
      .def(CopyableVisitor<ResidualModelState>());
}

}  // namespace python
}  // namespace crocoddyl
