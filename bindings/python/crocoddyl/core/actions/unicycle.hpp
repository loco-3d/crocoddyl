///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_PYTHON_CORE_ACTIONS_UNICYCLE_HPP_
#define CROCODDYL_PYTHON_CORE_ACTIONS_UNICYCLE_HPP_

#include <crocoddyl/core/actions/unicycle.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ActionModelUnicycle_wrap : public ActionModelUnicycle {
 public:
  ActionModelUnicycle_wrap() : ActionModelUnicycle() {}

  void calc_wrap1(std::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    calc(data, x, u);
  }

  void calc_wrap2(std::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x) { calc(data, x, unone_); }

  void calcDiff_wrap1(std::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                      bool recalc) {
    calcDiff(data, x, u, recalc);
  }

  void calcDiff_wrap2(std::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    calcDiff(data, x, u, true);
  }

  void calcDiff_wrap3(std::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x, bool recalc) {
    calcDiff(data, x, unone_, recalc);
  }

  void calcDiff_wrap4(std::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x, unone_, true);
  }
};

void exposeActionUnicycle() {
  bp::class_<ActionModelUnicycle_wrap, bp::bases<ActionModelAbstract>>(
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
      .def("calc", &ActionModelUnicycle_wrap::calc_wrap1, bp::args(" self", " data", " x", " u=None"),
           R"(Compute the next state and cost value.

It describes the time-discrete evolution of the unicycle system.
Additionally it computes the cost value associated to this discrete
state and control pair.
:param model: action model
:param data: action data
:param x: time-discrete state vector
:param u: time-discrete control input
:returns the next state and cost value)")
      .def("calc", &ActionModelUnicycle_wrap::calc_wrap2)
      .def("calcDiff", &ActionModelUnicycle_wrap::calcDiff_wrap1,
           bp::args(" self", " data", " x", " u=None", " recalc=True"),
           R"(Compute the derivatives of the unicycle dynamics and cost functions.

It computes the partial derivatives of the unicycle system and the
cost function. If recalc == True, it first updates the state evolution
and cost value. This function builds a quadratic approximation of the
action model (i.e. dynamical system and cost function).
:param model: action model
:param data: action data
:param x: time-discrete state vector
:param u: time-discrete control input
:param recalc: If true, it updates the state evolution and the cost value.
:returns the next state and cost value)")
      .def("calcDiff", &ActionModelUnicycle_wrap::calcDiff_wrap2)
      .def("calcDiff", &ActionModelUnicycle_wrap::calcDiff_wrap3)
      .def("calcDiff", &ActionModelUnicycle_wrap::calcDiff_wrap4)
      .def("createData", &ActionModelUnicycle_wrap::createData, bp::args(" self"),
           R"(Create the unicycle action data.)");

  boost::python::register_ptr_to_python<std::shared_ptr<ActionDataUnicycle>>();

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

#endif  // CROCODDYL_PYTHON_CORE_ACTIONS_UNICYCLE_HPP_