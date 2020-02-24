///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/actions/lqr.hpp"

namespace crocoddyl {
namespace python {

void exposeActionLQR() {
  bp::class_<ActionModelLQR, bp::bases<ActionModelAbstract> >(
      "ActionModelLQR",
      "LQR action model.\n\n"
      "A Linear-Quadratic Regulator problem has a transition model of the form\n"
      "xnext(x,u) = Fx*x + Fu*u + f0. Its cost function is quadratic of the\n"
      "form: 1/2 [x,u].T [Lxx Lxu; Lxu.T Luu] [x,u] + [lx,lu].T [x,u].",
      bp::init<int, int, bp::optional<bool> >(
          bp::args("self", "nx", "nu", "driftFree"),
          "Initialize the LQR action model.\n\n"
          ":param nx: dimension of the state vector\n"
          ":param nu: dimension of the control vector\n"
          ":param driftFree: enable/disable the bias term of the linear dynamics (default True)"))
      .def("calc", &ActionModelLQR::calc_wrap,
           ActionModel_calc_wraps(bp::args("self", "data", "x", "u"),
                                  "Compute the next state and cost value.\n\n"
                                  "It describes the time-discrete evolution of the LQR system. Additionally it\n"
                                  "computes the cost value associated to this discrete\n"
                                  "state and control pair.\n"
                                  ":param data: action data\n"
                                  ":param x: time-discrete state vector\n"
                                  ":param u: time-discrete control input"))
      .def<void (ActionModelLQR::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                    const Eigen::VectorXd&)>(
          "calcDiff", &ActionModelLQR::calcDiff_wrap, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the LQR dynamics and cost functions.\n\n"
          "It computes the partial derivatives of the LQR system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ActionModelLQR::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ActionModelLQR::calcDiff_wrap, bp::args("self", "data", "x"))
      .def("createData", &ActionModelLQR::createData, bp::args("self"), "Create the LQR action data.")
      .add_property("Fx", bp::make_function(&ActionModelLQR::get_Fx, bp::return_value_policy<bp::return_by_value>()),
                    &ActionModelLQR::set_Fx, "Jacobian of the dynamics")
      .add_property("Fu", bp::make_function(&ActionModelLQR::get_Fu, bp::return_value_policy<bp::return_by_value>()),
                    &ActionModelLQR::set_Fu, "Jacobian of the dynamics")
      .add_property("f0", bp::make_function(&ActionModelLQR::get_f0, bp::return_value_policy<bp::return_by_value>()),
                    &ActionModelLQR::set_f0, "dynamics drift")
      .add_property("lx", bp::make_function(&ActionModelLQR::get_lx, bp::return_value_policy<bp::return_by_value>()),
                    &ActionModelLQR::set_lx, "Jacobian of the cost")
      .add_property("lu", bp::make_function(&ActionModelLQR::get_lu, bp::return_value_policy<bp::return_by_value>()),
                    &ActionModelLQR::set_lu, "Jacobian of the cost")
      .add_property("Lxx", bp::make_function(&ActionModelLQR::get_Lxx, bp::return_value_policy<bp::return_by_value>()),
                    &ActionModelLQR::set_Lxx, "Hessian of the cost")
      .add_property("Lxu", bp::make_function(&ActionModelLQR::get_Lxu, bp::return_value_policy<bp::return_by_value>()),
                    &ActionModelLQR::set_Lxu, "Hessian of the cost")
      .add_property("Luu", bp::make_function(&ActionModelLQR::get_Luu, bp::return_value_policy<bp::return_by_value>()),
                    &ActionModelLQR::set_Luu, "Hessian of the cost");

  boost::python::register_ptr_to_python<boost::shared_ptr<ActionDataLQR> >();

  bp::class_<ActionDataLQR, bp::bases<ActionDataAbstract> >(
      "ActionDataLQR", "Action data for the LQR system.",
      bp::init<ActionModelLQR*>(bp::args("self", "model"),
                                "Create LQR data.\n\n"
                                ":param model: LQR action model"));
}

}  // namespace python
}  // namespace crocoddyl
