///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_

#include "crocoddyl/core/actions/unicycle.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActionUnicycle() {
  bp::class_<ActionModelUnicycle, bp::bases<ActionModelAbstract> >(
      "ActionModelUnicycle",
      "Unicycle action model.\n\n"
      "The transition model of an unicycle system is described as\n"
      "    xnext = [v*cos(theta); v*sin(theta); w],\n"
      "where the position is defined by (x, y, theta) and the control input\n"
      "by (v,w). Note that the state is defined only with the position. On the\n"
      "other hand, we define the quadratic cost functions for the state and\n"
      "control.",
      bp::init<>(bp::args("self"), "Initialize the unicycle action model."))
      .def("calc", &ActionModelUnicycle::calc_wrap,
           ActionModel_calc_wraps(bp::args("self", "data", "x", "u"),
                                  "Compute the next state and cost value.\n\n"
                                  "It describes the time-discrete evolution of the unicycle system.\n"
                                  "Additionally it computes the cost value associated to this discrete\n"
                                  "state and control pair.\n"
                                  ":param data: action data\n"
                                  ":param x: time-discrete state vector\n"
                                  ":param u: time-discrete control input"))
      .def<void (ActionModelUnicycle::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                         const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &ActionModelUnicycle::calcDiff_wrap, bp::args("self", "data", "x", "u", "recalc"),
          "Compute the derivatives of the unicycle dynamics and cost functions.\n\n"
          "It computes the partial derivatives of the unicycle system and the\n"
          "cost function. If recalc == True, it first updates the state evolution\n"
          "and cost value. This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value (default True).")
      .def<void (ActionModelUnicycle::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                         const Eigen::VectorXd&)>("calcDiff", &ActionModelUnicycle::calcDiff_wrap,
                                                                  bp::args("self", "data", "x", "u"))
      .def<void (ActionModelUnicycle::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ActionModelUnicycle::calcDiff_wrap, bp::args("self", "data", "x"))
      .def<void (ActionModelUnicycle::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                         const bool&)>("calcDiff", &ActionModelUnicycle::calcDiff_wrap,
                                                       bp::args("self", "data", "x", "recalc"))
      .def("createData", &ActionModelUnicycle::createData, bp::args("self"), "Create the unicycle action data.")
      .add_property(
          "costWeights",
          bp::make_function(&ActionModelUnicycle::get_cost_weights, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ActionModelUnicycle::set_cost_weights), "cost weights");

  bp::register_ptr_to_python<boost::shared_ptr<ActionDataUnicycle> >();

  bp::class_<ActionDataUnicycle, bp::bases<ActionDataAbstract> >(
      "ActionDataUnicycle",
      "Action data for the Unicycle system.\n\n"
      "The unicycle data, apart of common one, contains the cost residuals used\n"
      "for the computation of calc and calcDiff.",
      bp::init<ActionModelUnicycle*>(bp::args("self", "model"),
                                     "Create unicycle data.\n\n"
                                     ":param model: unicycle action model"));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_
