///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_ACTIONS_ACTION_UNICYCLE_HPP_
#define PYTHON_CROCODDYL_CORE_ACTIONS_ACTION_UNICYCLE_HPP_

#include "crocoddyl/core/actions/action-unicycle.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActionUnicycle() {
  bp::class_<ActionModelUnicycle, bp::bases<ActionModelAbstract>>(
      "ActionModelUnicycle",
      R"(Unicycle action model.

        The transition model of an unicycle system is described as
            xnext = [v*cos(theta); v*sin(theta); w],
        where the position is defined by (x, y, theta) and the control input
        by (v,w). Note that the state is defined only with the position. On the
        other hand, we define the quadratic cost functions for the state and
        control.)",
      bp::init<>(bp::args(" self"),
                 R"(Initialize the unicycle action model.)"))
      .def<void (ActionModelUnicycle::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                         const Eigen::VectorXd&)>("calc", &ActionModelUnicycle::calc_wrap,
                                                                  bp::args(" self", " data", " x", " u=None"),
                                                                  R"(Compute the next state and cost value.

It describes the time-discrete evolution of the unicycle system.
Additionally it computes the cost value associated to this discrete
state and control pair.
:param data: action data
:param x: time-discrete state vector
:param u: time-discrete control input)")
      .def<void (ActionModelUnicycle::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calc", &ActionModelUnicycle::calc_wrap, bp::args(" self", " data", " x"))
      .def<void (ActionModelUnicycle::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                         const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &ActionModelUnicycle::calcDiff_wrap, bp::args(" self", " data", " x", " u=None", " recalc=True"),
          R"(Compute the derivatives of the unicycle dynamics and cost functions.

It computes the partial derivatives of the unicycle system and the
cost function. If recalc == True, it first updates the state evolution
and cost value. This function builds a quadratic approximation of the
action model (i.e. dynamical system and cost function).
:param data: action data
:param x: time-discrete state vector
:param u: time-discrete control input
:param recalc: If true, it updates the state evolution and the cost value.)")
      .def<void (ActionModelUnicycle::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                         const Eigen::VectorXd&)>("calcDiff", &ActionModelUnicycle::calcDiff_wrap,
                                                                  bp::args(" self", " data", " x", " u"))
      .def<void (ActionModelUnicycle::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ActionModelUnicycle::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (ActionModelUnicycle::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &ActionModelUnicycle::calcDiff_wrap, bp::args(" self", " data", " x", " recalc"))
      .def("createData", &ActionModelUnicycle::createData, bp::args(" self"),
           R"(Create the unicycle action data.)");

  bp::register_ptr_to_python<std::shared_ptr<ActionDataUnicycle>>();

  bp::class_<ActionDataUnicycle, bp::bases<ActionDataAbstract>>(
      "ActionDataUnicycle",
      R"(Action data for the Unicycle system.

        The unicycle data, apart of common one, contains the cost residuals used
        for the computation of calc and calcDiff.)",
      bp::init<ActionModelUnicycle*>(bp::args(" self", " model"),
                                     R"(Create unicycle data.

:param model: unicycle action model)"));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_ACTIONS_ACTION_UNICYCLE_HPP_