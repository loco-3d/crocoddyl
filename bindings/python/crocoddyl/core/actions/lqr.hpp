///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_ACTIONS_LQR_HPP_
#define PYTHON_CROCODDYL_CORE_ACTIONS_LQR_HPP_

#include "crocoddyl/core/actions/lqr.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActionLQR() {
  bp::class_<ActionModelLQR, bp::bases<ActionModelAbstract> >(
      "ActionModelLQR",
      "LQR action model.\n\n"
      "A Linear-Quadratic Regulator problem has a transition model of the form\n"
      "xnext(x,u) = Fx*x + Fu*u + f0. Its cost function is quadratic of the\n"
      "form: 1/2 [x,u].T [Lxx Lxu; Lxu.T Luu] [x,u] + [lx,lu].T [x,u].",
      bp::init<int, int, bp::optional<bool> >(bp::args(" self", " nx", " ndu", " driftFree=True"),
                                              "Initialize the LQR action model.\n\n"
                                              ":param nx: dimension of the state vector\n"
                                              ":param nu: dimension of the control vector\n"
                                              ":param driftFree: enable/disable the bias term of the linear dynamics"))
      .def<void (ActionModelLQR::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                    const Eigen::VectorXd&)>(
          "calc", &ActionModelLQR::calc_wrap, bp::args(" self", " data", " x", " u=None"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-discrete evolution of the LQR system. Additionally it\n"
          "computes the cost value associated to this discrete\n"
          "state and control pair.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ActionModelLQR::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calc", &ActionModelLQR::calc_wrap, bp::args(" self", " data", " x"))
      .def<void (ActionModelLQR::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                    const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &ActionModelLQR::calcDiff_wrap, bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the LQR dynamics and cost functions.\n\n"
          "It computes the partial derivatives of the LQR system and the\n"
          "cost function. If recalc == True, it first updates the state evolution\n"
          "and cost value. This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (ActionModelLQR::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                    const Eigen::VectorXd&)>("calcDiff", &ActionModelLQR::calcDiff_wrap,
                                                             bp::args(" self", " data", " x", " u"))
      .def<void (ActionModelLQR::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ActionModelLQR::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (ActionModelLQR::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                    const bool&)>("calcDiff", &ActionModelLQR::calcDiff_wrap,
                                                  bp::args(" self", " data", " x", " recalc"))
      .def("createData", &ActionModelLQR::createData, bp::args(" self"), "Create the LQR action data.");

  boost::python::register_ptr_to_python<boost::shared_ptr<ActionDataLQR> >();

  bp::class_<ActionDataLQR, bp::bases<ActionDataAbstract> >(
      "ActionDataLQR", "Action data for the LQR system.",
      bp::init<ActionModelLQR*>(bp::args(" self", " model"),
                                "Create LQR data.\n\n"
                                ":param model: LQR action model"));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_ACTIONS_LQR_HPP_