///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_ACTIONS_LQR_HPP_
#define PYTHON_CROCODDYL_CORE_ACTIONS_LQR_HPP_

#include <crocoddyl/core/actions/lqr.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ActionModelLQR_wrap : public ActionModelLQR {
 public:
  ActionModelLQR_wrap(int nx, int nu, bool drift_free = true) : ActionModelLQR(nx, nu, drift_free) {}

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

void exposeActionLQR() {
  bp::class_<ActionModelLQR_wrap, bp::bases<ActionModelAbstract>>(
      "ActionModelLQR",
      R"(LQR action model.

        A Linear-Quadratic Regulator problem has a transition model of the form
        xnext(x,u) = Fx*x + Fu*u + f0. Its cost function is quadratic of the
        form: 1/2 [x,u].T [Lxx Lxu; Lxu.T Luu] [x,u] + [lx,lu].T [x,u].)",
      bp::init<int, int, bp::optional<bool>>(bp::args(" self", " nx", " ndu", " driftFree=True"),
                                             R"(Initialize the LQR action model.

:param nx: dimension of the state vector
:param nu: dimension of the control vector
:param driftFree: enable/disable the bias term of the linear dynamics)"))
      .def("calc", &ActionModelLQR_wrap::calc_wrap1, bp::args(" self", " data", " x", " u=None"),
           R"(Compute the next state and cost value.

It describes the time-discrete evolution of the LQR system. Additionally it
computes the cost value associated to this discrete
state and control pair.
:param data: action data
:param x: time-discrete state vector
:param u: time-discrete control input
:returns the next state and cost value)")
      .def("calc", &ActionModelLQR_wrap::calc_wrap2)
      .def("calcDiff", &ActionModelLQR_wrap::calcDiff_wrap1,
           bp::args(" self", " data", " x", " u=None", " recalc=True"),
           R"(Compute the derivatives of the LQR dynamics and cost functions.

It computes the partial derivatives of the LQR system and the
cost function. If recalc == True, it first updates the state evolution
and cost value. This function builds a quadratic approximation of the
action model (i.e. dynamical system and cost function).
:param data: action data
:param x: time-discrete state vector
:param u: time-discrete control input
:param recalc: If true, it updates the state evolution and the cost value.
:returns the next state and cost value)")
      .def("calcDiff", &ActionModelLQR_wrap::calcDiff_wrap2)
      .def("calcDiff", &ActionModelLQR_wrap::calcDiff_wrap3)
      .def("calcDiff", &ActionModelLQR_wrap::calcDiff_wrap4)
      .def("createData", &ActionModelLQR_wrap::createData, bp::args(" self"),
           R"(Create the LQR action data.)");

  boost::python::register_ptr_to_python<std::shared_ptr<ActionDataLQR>>();

  bp::class_<ActionDataLQR, bp::bases<ActionDataAbstract>>("ActionDataLQR",
                                                           R"(Action data for the LQR system.

        The unicicycle data, apart of common one, contains the cost residuals used
        for the computation of calc and calcDiff.)",
                                                           bp::init<ActionModelLQR*>(bp::args(" self", " model"),
                                                                                     R"(Create LQR data.

:param model: LQR action model)"));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_ACTIONS_LQR_HPP_