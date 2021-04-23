///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"

namespace crocoddyl {
namespace python {

void exposeActionUnicycle() {
  bp::register_ptr_to_python<boost::shared_ptr<ActionModelUnicycle> >();

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
      .def<void (ActionModelUnicycle::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelUnicycle::calc, bp::args("self", "data", "x", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-discrete evolution of the unicycle system.\n"
          "Additionally it computes the cost value associated to this discrete\n"
          "state and control pair.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ActionModelUnicycle::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &ActionModelAbstract::calc,
                                                                                    bp::args("self", "data", "x"))
      .def<void (ActionModelUnicycle::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelUnicycle::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the unicycle dynamics and cost functions.\n\n"
          "It computes the partial derivatives of the unicycle system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ActionModelUnicycle::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ActionModelUnicycle::createData, bp::args("self"), "Create the unicycle action data.")
      .add_property("dt", bp::make_function(&ActionModelUnicycle::get_dt),
                    bp::make_function(&ActionModelUnicycle::set_dt), "integration time")
      .add_property("costWeights",
                    bp::make_function(&ActionModelUnicycle::get_cost_weights, bp::return_internal_reference<>()),
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
