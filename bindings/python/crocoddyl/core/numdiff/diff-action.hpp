///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_

#include "crocoddyl/core/numdiff/diff-action.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeDifferentialActionNumDiff() {
  bp::class_<DifferentialActionModelNumDiff, bp::bases<DifferentialActionModelAbstract> >(
      "DifferentialActionModelNumDiff",
      "Abstract class for computing calcDiff by using numerical differentiation.\n\n",
      bp::init<DifferentialActionModelAbstract&, bp::optional<bool> >(
          bp::args(" self", " model", " gaussApprox=False"),
          "Initialize the action model NumDiff.\n\n"
          ":param model: action model where we compute the derivatives through NumDiff,\n"
          ":param gaussApprox: compute the Hessian using Gauss approximation")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", &DifferentialActionModelNumDiff::calc_wrap,
           DiffActionModel_calc_wraps(bp::args(" self", " data", " x", " u=None"),
                                      "Compute the next state and cost value.\n\n"
                                      "The system evolution is described in model.\n"
                                      ":param data: NumDiff action data\n"
                                      ":param x: time-discrete state vector\n"
                                      ":param u: time-discrete control input"))
      .def<void (DifferentialActionModelNumDiff::*)(const boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                    const Eigen::VectorXd&, const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &DifferentialActionModelNumDiff::calcDiff_wrap,
          bp::args(" self", " data", " x", " u", " recalc=True"),
          "Compute the derivatives of the dynamics and cost functions.\n\n"
          "It computes the Jacobian and Hessian using numerical differentiation.\n"
          ":param data: NumDiff action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (DifferentialActionModelNumDiff::*)(const boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                    const Eigen::VectorXd&, const Eigen::VectorXd&)>(
          "calcDiff", &DifferentialActionModelNumDiff::calcDiff_wrap, bp::args(" self", " data", " x", " u"))
      .def<void (DifferentialActionModelNumDiff::*)(const boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                    const Eigen::VectorXd&)>(
          "calcDiff", &DifferentialActionModelNumDiff::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (DifferentialActionModelNumDiff::*)(const boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                    const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &DifferentialActionModelNumDiff::calcDiff_wrap, bp::args(" self", " data", " x", " recalc"))
      .def("createData", &DifferentialActionModelNumDiff::createData, bp::args(" self"),
           "Create the action data.\n\n"
           "Each action model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property("model",
                    bp::make_function(&DifferentialActionModelNumDiff::get_model, bp::return_internal_reference<>()),
                    "action model");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
