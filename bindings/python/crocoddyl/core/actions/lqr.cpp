///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/lqr.hpp"

#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeActionLQR() {
// TODO: Remove once the deprecated update call has been removed in a future
// release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  boost::python::register_ptr_to_python<boost::shared_ptr<ActionModelLQR> >();

  bp::class_<ActionModelLQR, bp::bases<ActionModelAbstract> >(
      "ActionModelLQR",
      "LQR action model.\n\n"
      "A Linear-Quadratic Regulator problem has a transition model of the "
      "form\n"
      "xnext(x,u) = Fx*x + Fu*u + f0. Its cost function is quadratic of the\n"
      "form: 1/2 [x,u].T [Lxx Lxu; Lxu.T Luu] [x,u] + [lx,lu].T [x,u].",
      bp::init<int, int, bp::optional<bool> >(
          bp::args("self", "nx", "nu", "driftFree"),
          "Initialize the LQR action model.\n\n"
          ":param nx: dimension of the state vector\n"
          ":param nu: dimension of the control vector\n"
          ":param driftFree: enable/disable the bias term of the linear "
          "dynamics (default True)"))
      .def<void (ActionModelLQR::*)(
          const boost::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelLQR::calc, bp::args("self", "data", "x", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-discrete evolution of the LQR system. "
          "Additionally it\n"
          "computes the cost value associated to this discrete\n"
          "state and control pair.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ActionModelLQR::*)(
          const boost::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ActionModelLQR::*)(
          const boost::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelLQR::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the LQR dynamics and cost functions.\n\n"
          "It computes the partial derivatives of the LQR system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ActionModelLQR::*)(
          const boost::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ActionModelLQR::createData, bp::args("self"),
           "Create the LQR action data.")
      .add_property("A",
                    bp::make_function(&ActionModelLQR::get_A,
                                      bp::return_internal_reference<>()),
                    &ActionModelLQR::set_A, "state matrix")
      .add_property("B",
                    bp::make_function(&ActionModelLQR::get_B,
                                      bp::return_internal_reference<>()),
                    &ActionModelLQR::set_B, "input matrix")
      .add_property("f",
                    bp::make_function(&ActionModelLQR::get_f,
                                      bp::return_internal_reference<>()),
                    &ActionModelLQR::set_f, "dynamics drift")
      .add_property("Q",
                    bp::make_function(&ActionModelLQR::get_Q,
                                      bp::return_internal_reference<>()),
                    &ActionModelLQR::set_Q, "state weight matrix")
      .add_property("R",
                    bp::make_function(&ActionModelLQR::get_R,
                                      bp::return_internal_reference<>()),
                    &ActionModelLQR::set_R, "input weight matrix")
      .add_property("N",
                    bp::make_function(&ActionModelLQR::get_N,
                                      bp::return_internal_reference<>()),
                    &ActionModelLQR::set_N, "state-input weight matrix")
      .add_property("q",
                    bp::make_function(&ActionModelLQR::get_q,
                                      bp::return_internal_reference<>()),
                    &ActionModelLQR::set_q, "state weight vector")
      .add_property("r",
                    bp::make_function(&ActionModelLQR::get_r,
                                      bp::return_internal_reference<>()),
                    &ActionModelLQR::set_r, "input weight vector")
      // deprecated function
      .add_property(
          "Fx",
          bp::make_function(&ActionModelLQR::get_A,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use A.")),
          &ActionModelLQR::set_A, "state matrix")
      .add_property(
          "Fu",
          bp::make_function(&ActionModelLQR::get_B,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use B.")),
          bp::make_function(&ActionModelLQR::set_B,
                            deprecated<>("Deprecated. Use B.")),
          "input matrix")
      .add_property(
          "f0",
          bp::make_function(&ActionModelLQR::get_f,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use f.")),
          bp::make_function(&ActionModelLQR::set_f,
                            deprecated<>("Deprecated. Use f.")),
          "dynamics drift")
      .add_property(
          "lx",
          bp::make_function(&ActionModelLQR::get_q,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use q.")),
          bp::make_function(&ActionModelLQR::set_q,
                            deprecated<>("Deprecated. Use q.")),
          "state weight vector")
      .add_property(
          "lu",
          bp::make_function(&ActionModelLQR::get_r,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use r.")),
          bp::make_function(&ActionModelLQR::set_r,
                            deprecated<>("Deprecated. Use r.")),
          "input weight vector")
      .add_property(
          "Lxx",
          bp::make_function(&ActionModelLQR::get_Q,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use Q.")),
          bp::make_function(&ActionModelLQR::set_Q,
                            deprecated<>("Deprecated. Use Q.")),
          "state weight matrix")
      .add_property(
          "Lxu",
          bp::make_function(&ActionModelLQR::get_N,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use N.")),
          bp::make_function(&ActionModelLQR::set_N,
                            deprecated<>("Deprecated. Use N.")),
          "state-input weight matrix")
      .add_property(
          "Luu",
          bp::make_function(&ActionModelLQR::get_R,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use R.")),
          bp::make_function(&ActionModelLQR::set_R,
                            deprecated<>("Deprecated. Use R.")),
          "input weight matrix")
      .def(CopyableVisitor<ActionModelLQR>());

  boost::python::register_ptr_to_python<boost::shared_ptr<ActionDataLQR> >();

  bp::class_<ActionDataLQR, bp::bases<ActionDataAbstract> >(
      "ActionDataLQR", "Action data for the LQR system.",
      bp::init<ActionModelLQR*>(bp::args("self", "model"),
                                "Create LQR data.\n\n"
                                ":param model: LQR action model"))
      .def(CopyableVisitor<ActionDataLQR>());

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
