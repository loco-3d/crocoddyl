///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {
namespace python {

void exposeActionNumDiff() {
  bp::class_<ActionModelNumDiff, bp::bases<ActionModelAbstract> >(
      "ActionModelNumDiff", "Abstract class for computing calcDiff by using numerical differentiation.\n\n",
      bp::init<boost::shared_ptr<ActionModelAbstract> >(
          bp::args("self", "model"),
          "Initialize the action model NumDiff.\n\n"
          ":param model: action model where we compute the derivatives through NumDiff"))
      .def("calc", &ActionModelNumDiff::calc_wrap,
           ActionModel_calc_wraps(bp::args("self", "data", "x", "u"),
                                  "Compute the next state and cost value.\n\n"
                                  "The system evolution is described in model.\n"
                                  ":param data: NumDiff action data\n"
                                  ":param x: time-discrete state vector\n"
                                  ":param u: time-discrete control input"))
      .def<void (ActionModelNumDiff::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                        const Eigen::VectorXd&)>(
          "calcDiff", &ActionModelNumDiff::calcDiff_wrap, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the dynamics and cost functions.\n\n"
          "It computes the Jacobian and Hessian using numerical differentiation.\n"
          ":param data: NumDiff action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ActionModelNumDiff::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ActionModelNumDiff::calcDiff_wrap, bp::args("self", "data", "x"))
      .def("createData", &ActionModelNumDiff::createData, bp::args("self"),
           "Create the action data.\n\n"
           "Each action model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property("model",
                    bp::make_function(&ActionModelNumDiff::get_model, bp::return_value_policy<bp::return_by_value>()),
                    "action model")
      .add_property(
          "disturbance",
          bp::make_function(&ActionModelNumDiff::get_disturbance, bp::return_value_policy<bp::return_by_value>()),
          &ActionModelNumDiff::set_disturbance, "disturbance value used in the numerical differentiation")
      .add_property("withGaussApprox",
                    bp::make_function(&ActionModelNumDiff::get_with_gauss_approx,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "Gauss approximation for computing the Hessians");

  bp::register_ptr_to_python<boost::shared_ptr<ActionDataNumDiff> >();

  bp::class_<ActionDataNumDiff, bp::bases<ActionDataAbstract> >(
      "ActionDataNumDiff", "Numerical differentiation action data.",
      bp::init<ActionModelNumDiff*>(bp::args("self", "model"),
                                    "Create numerical differentiation action data.\n\n"
                                    ":param model: numdiff action model"))
      .add_property("Rx", bp::make_getter(&ActionDataNumDiff::Rx, bp::return_value_policy<bp::return_by_value>()),
                    "Jacobian of the cost residual.")
      .add_property("Ru", bp::make_getter(&ActionDataNumDiff::Ru, bp::return_value_policy<bp::return_by_value>()),
                    "Jacobian of the cost residual.")
      .add_property("dx", bp::make_getter(&ActionDataNumDiff::dx, bp::return_value_policy<bp::return_by_value>()),
                    "state disturbance.")
      .add_property("du", bp::make_getter(&ActionDataNumDiff::du, bp::return_value_policy<bp::return_by_value>()),
                    "control disturbance.")
      .add_property("data_0",
                    bp::make_getter(&ActionDataNumDiff::data_0, bp::return_value_policy<bp::return_by_value>()),
                    "data that contains the final results")
      .add_property("data_x",
                    bp::make_getter(&ActionDataNumDiff::data_x, bp::return_value_policy<bp::return_by_value>()),
                    "temporary data associated with the state variation")
      .add_property("data_u",
                    bp::make_getter(&ActionDataNumDiff::data_u, bp::return_value_policy<bp::return_by_value>()),
                    "temporary data associated with the control variation");
}

}  // namespace python
}  // namespace crocoddyl
