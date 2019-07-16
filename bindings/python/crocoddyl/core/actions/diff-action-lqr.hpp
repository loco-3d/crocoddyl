///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_ACTIONS_DIFF_ACTION_LQR_HPP_
#define PYTHON_CROCODDYL_CORE_ACTIONS_DIFF_ACTION_LQR_HPP_

#include "crocoddyl/core/actions/diff-action-lqr.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class DifferentialActionModelLQR_wrap : public DifferentialActionModelLQR {
 public:
  DifferentialActionModelLQR_wrap(int nx, int nu, bool drift_free = true)
      : DifferentialActionModelLQR(nx, nu, drift_free) {
    // We need to change it to the wrap object in Python
    state_->~StateAbstract();                    // destroy the object but leave the space allocated
    state_ = new (state_) StateVector_wrap(nx);  // create a new object in the same space
  }

  void calc_wrap1(std::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x,
                  const Eigen::VectorXd& u) {
    calc(data, x, u);
  }

  void calc_wrap2(std::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x) {
    calc(data, x, unone_);
  }

  void calcDiff_wrap1(std::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u, bool recalc) {
    calcDiff(data, x, u, recalc);
  }

  void calcDiff_wrap2(std::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u) {
    calcDiff(data, x, u, true);
  }

  void calcDiff_wrap3(std::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x, bool recalc) {
    calcDiff(data, x, unone_, recalc);
  }

  void calcDiff_wrap4(std::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x, unone_, true);
  }
};

void exposeDifferentialActionLQR() {
  bp::class_<DifferentialActionModelLQR_wrap, bp::bases<DifferentialActionModelAbstract>>(
      "DifferentialActionModelLQR",
      R"(Differential action model for linear dynamics and quadratic cost.

        This class implements a linear dynamics, and quadratic costs (i.e.
        LQR action). Since the DAM is a second order system, and the integrated
        action models are implemented as being second order integrators. This
        class implements a second order linear system given by
          x = [q, v]
          dv = Fq q + Fv v + Fu u + f0
        where Fq, Fv, Fu and f0 are randomly chosen constant terms. On the other
        hand the cost function is given by
          l(x,u) = 1/2 [x,u].T [Lxx Lxu; Lxu.T Luu] [x,u] + [lx,lu].T [x,u].)",
      bp::init<int, int, bp::optional<bool>>(bp::args(" self", " nx", " ndu", " driftFree=True"),
                                             R"(Initialize the differential LQR action model.

:param nx: dimension of the state vector
:param nu: dimension of the control vector
:param driftFree: enable/disable the bias term of the linear dynamics)"))
      .def("calc", &DifferentialActionModelLQR_wrap::calc_wrap1, bp::args(" self", " data", " x", " u=None"),
           R"(Compute the next state and cost value.

It describes the time-continuous evolution of the LQR system. Additionally it
computes the cost value associated to this discrete
state and control pair.
:param data: action data
:param x: time-continuous state vector
:param u: time-continuous control input)")
      .def("calc", &DifferentialActionModelLQR_wrap::calc_wrap2)
      .def("calcDiff", &DifferentialActionModelLQR_wrap::calcDiff_wrap1,
           bp::args(" self", " data", " x", " u=None", " recalc=True"),
           R"(Compute the derivatives of the differential LQR dynamics and cost functions.

It computes the partial derivatives of the differential LQR system and the
cost function. If recalc == True, it first updates the state evolution
and cost value. This function builds a quadratic approximation of the
action model (i.e. dynamical system and cost function).
:param data: action data
:param x: time-continuous state vector
:param u: time-continuous control input
:param recalc: If true, it updates the state evolution and the cost value.)")
      .def("calcDiff", &DifferentialActionModelLQR_wrap::calcDiff_wrap2)
      .def("calcDiff", &DifferentialActionModelLQR_wrap::calcDiff_wrap3)
      .def("calcDiff", &DifferentialActionModelLQR_wrap::calcDiff_wrap4)
      .def("createData", &DifferentialActionModelLQR_wrap::createData, bp::args(" self"),
           R"(Create the differential LQR action data.)");

  boost::python::register_ptr_to_python<std::shared_ptr<DifferentialActionDataLQR>>();

  bp::class_<DifferentialActionDataLQR, bp::bases<DifferentialActionDataAbstract>>(
      "DifferentialActionDataLQR",
      R"(Action data for the differential LQR system.)",
      bp::init<DifferentialActionModelLQR*>(bp::args(" self", " model"),
                                            R"(Create differential LQR data.

:param model: differential LQR action model)"));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_ACTIONS_DIFF_ACTION_LQR_HPP_