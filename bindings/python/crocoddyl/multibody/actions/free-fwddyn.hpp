///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_ACTIONS_DIFF_FREE_FWDDYN_HPP_
#define PYTHON_CROCODDYL_CORE_ACTIONS_DIFF_FREE_FWDDYN_HPP_

#include "crocoddyl/multibody/actions/free-fwddyn.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeDifferentialActionFreeFwdDynamics() {
  bp::class_<DifferentialActionModelFreeFwdDynamics, bp::bases<DifferentialActionModelAbstract> >(
      "DifferentialActionModelFreeFwdDynamics",
      // "Differential action model for linear dynamics and quadratic cost.\n\n"
      // "This class implements a linear dynamics, and quadratic costs (i.e.\n"
      // "LQR action). Since the DAM is a second order system, and the integrated\n"
      // "action models are implemented as being second order integrators. This\n"
      // "class implements a second order linear system given by\n"
      // "  x = [q, v]\n"
      // "  dv = Fq q + Fv v + Fu u + f0\n"
      // "where Fq, Fv, Fu and f0 are randomly chosen constant terms. On the other\n"
      // "hand the cost function is given by\n"
      // "  l(x,u) = 1/2 [x,u].T [Lxx Lxu; Lxu.T Luu] [x,u] + [lx,lu].T [x,u].",
      bp::init<StateMultibody*, CostModelSum*>(
          bp::args(" self", " state", " costs"),
          "Initialize the free forward-dynamics action model.\n\n"
          ":param state: multibody state\n"
          ":param costs: stack of cost functions")[bp::with_custodian_and_ward<1, 3>()])
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                            const Eigen::VectorXd&, const Eigen::VectorXd&)>(
          "calc", &DifferentialActionModelFreeFwdDynamics::calc_wrap, bp::args(" self", " data", " x", " u=None"),
          "Compute the next state and cost value.\n\n"
          // "It describes the time-continuous evolution of the LQR system. Additionally it\n"
          // "computes the cost value associated to this discrete state and control pair.\n"
          ":param data: free forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input")
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                            const Eigen::VectorXd&)>(
          "calc", &DifferentialActionModelFreeFwdDynamics::calc_wrap, bp::args(" self", " data", " x"))
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(
          boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::VectorXd&, const Eigen::VectorXd&,
          const bool&)>("calcDiff", &DifferentialActionModelFreeFwdDynamics::calcDiff_wrap,
                        bp::args(" self", " data", " x", " u=None", " recalc=True"),
                        // "Compute the derivatives of the differential LQR dynamics and cost functions.\n\n"
                        // "It computes the partial derivatives of the differential LQR system and the\n"
                        // "cost function. If recalc == True, it first updates the state evolution\n"
                        // "and cost value. This function builds a quadratic approximation of the\n"
                        // "action model (i.e. dynamical system and cost function).\n"
                        ":param data: free forward-dynamics action data\n"
                        ":param x: time-continuous state vector\n"
                        ":param u: time-continuous control input\n"
                        ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                            const Eigen::VectorXd&, const Eigen::VectorXd&)>(
          "calcDiff", &DifferentialActionModelFreeFwdDynamics::calcDiff_wrap, bp::args(" self", " data", " x", " u"))
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                            const Eigen::VectorXd&)>(
          "calcDiff", &DifferentialActionModelFreeFwdDynamics::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                            const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &DifferentialActionModelFreeFwdDynamics::calcDiff_wrap,
          bp::args(" self", " data", " x", " recalc"))
      .def("createData", &DifferentialActionModelFreeFwdDynamics::createData, bp::args(" self"),
           "Create the differential LQR action data.");

  boost::python::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataFreeFwdDynamics> >();

  bp::class_<DifferentialActionDataFreeFwdDynamics, bp::bases<DifferentialActionDataAbstract> >(
      "DifferentialActionDataLQR", "Action data for the differential LQR system.",
      bp::init<DifferentialActionModelFreeFwdDynamics*>(bp::args(" self", " model"),
                                                        "Create free forward-dynamics action data.\n\n"
                                                        ":param model: free forward-dynamics action model"));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_ACTIONS_DIFF_FREE_FWDDYN_HPP_