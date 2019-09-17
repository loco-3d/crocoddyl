///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_ACTION_HPP_

#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActionNumDiff() {
  bp::class_<ActionModelNumDiff, bp::bases<ActionModelAbstract> >(
      "ActionModelNumDiff", "Abstract class for computing calcDiff by using numerical differentiation.\n\n",
      bp::init<ActionModelAbstract&, bp::optional<bool> >(
          bp::args(" self", " model", " gaussApprox=False"),
          "Initialize the action model NumDiff.\n\n"
          ":param model: action model where we compute the derivatives through NumDiff,\n"
          ":param gaussApprox: compute the Hessian using Gauss approximation")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", &ActionModelNumDiff::calc_wrap,
           ActionModel_calc_wraps(bp::args(" self", " data", " x", " u=None"),
                                  "Compute the next state and cost value.\n\n"
                                  "The system evolution is described in model.\n"
                                  ":param data: NumDiff action data\n"
                                  ":param x: time-discrete state vector\n"
                                  ":param u: time-discrete control input"))
      .def<void (ActionModelNumDiff::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                        const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &ActionModelNumDiff::calcDiff_wrap, bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the dynamics and cost functions.\n\n"
          "It computes the Jacobian and Hessian using numerical differentiation.\n"
          ":param data: NumDiff action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (ActionModelNumDiff::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                        const Eigen::VectorXd&)>("calcDiff", &ActionModelNumDiff::calcDiff_wrap,
                                                                 bp::args(" self", " data", " x", " u"))
      .def<void (ActionModelNumDiff::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ActionModelNumDiff::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (ActionModelNumDiff::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                        const bool&)>("calcDiff", &ActionModelNumDiff::calcDiff_wrap,
                                                      bp::args(" self", " data", " x", " recalc"))
      .def("createData", &ActionModelNumDiff::createData, bp::args(" self"),
           "Create the action data.\n\n"
           "Each action model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property("model", bp::make_function(&ActionModelNumDiff::get_model, bp::return_internal_reference<>()),
                    "action model")
      .add_property(
          "disturbance",
          bp::make_function(&ActionModelNumDiff::get_disturbance, bp::return_value_policy<bp::return_by_value>()),
          "disturbance value used in the numerical differentiation")
      .add_property("withGaussApprox",
                    bp::make_function(&ActionModelNumDiff::get_with_gauss_approx,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "Gauss approximation for computing the Hessians");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
