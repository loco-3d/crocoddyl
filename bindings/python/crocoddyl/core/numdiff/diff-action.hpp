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
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeDifferentialActionNumDiff() {
  // Register custom converters between std::vector and Python list
  typedef boost::shared_ptr<DifferentialActionModelAbstract> DifferentialActionModelPtr;
  typedef boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionDataPtr;
  bp::to_python_converter<std::vector<DifferentialActionModelPtr, std::allocator<DifferentialActionModelPtr> >,
                          vector_to_list<DifferentialActionModelPtr, false> >();
  bp::to_python_converter<std::vector<DifferentialActionDataPtr, std::allocator<DifferentialActionDataPtr> >,
                          vector_to_list<DifferentialActionDataPtr, false> >();
  list_to_vector()
      .from_python<std::vector<DifferentialActionModelPtr, std::allocator<DifferentialActionModelPtr> > >();

  bp::class_<DifferentialActionModelNumDiff, bp::bases<DifferentialActionModelAbstract> >(
      "DifferentialActionModelNumDiff",
      "Abstract class for computing calcDiff by using numerical differentiation.\n\n",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<bool> >(
          bp::args("self", "model", "gaussApprox"),
          "Initialize the action model NumDiff.\n\n"
          ":param model: action model where we compute the derivatives through NumDiff,\n"
          ":param gaussApprox: compute the Hessian using Gauss approximation (default False)"))
      .def("calc", &DifferentialActionModelNumDiff::calc_wrap,
           DiffActionModel_calc_wraps(bp::args("self", "data", "x", "u"),
                                      "Compute the next state and cost value.\n\n"
                                      "The system evolution is described in model.\n"
                                      ":param data: NumDiff action data\n"
                                      ":param x: time-discrete state vector\n"
                                      ":param u: time-discrete control input"))
      .def<void (DifferentialActionModelNumDiff::*)(const boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                    const Eigen::VectorXd&, const Eigen::VectorXd&)>(
          "calcDiff", &DifferentialActionModelNumDiff::calcDiff_wrap, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the dynamics and cost functions.\n\n"
          "It computes the Jacobian and Hessian using numerical differentiation.\n"
          ":param data: NumDiff action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (DifferentialActionModelNumDiff::*)(const boost::shared_ptr<DifferentialActionDataAbstract>&,
                                                    const Eigen::VectorXd&)>(
          "calcDiff", &DifferentialActionModelNumDiff::calcDiff_wrap, bp::args("self", "data", "x"))
      .def("createData", &DifferentialActionModelNumDiff::createData, bp::args("self"),
           "Create the action data.\n\n"
           "Each action model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property("model",
                    bp::make_function(&DifferentialActionModelNumDiff::get_model,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "action model")
      .add_property("disturbance",
                    bp::make_function(&DifferentialActionModelNumDiff::get_disturbance,
                                      bp::return_value_policy<bp::return_by_value>()),
                    &DifferentialActionModelNumDiff::set_disturbance,
                    "disturbance value used in the numerical differentiation")
      .add_property("withGaussApprox",
                    bp::make_function(&DifferentialActionModelNumDiff::get_with_gauss_approx,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "Gauss approximation for computing the Hessians");

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataNumDiff> >();

  bp::class_<DifferentialActionDataNumDiff, bp::bases<DifferentialActionDataAbstract> >(
      "DifferentialActionDataNumDiff", "Numerical differentiation diff-action data.",
      bp::init<DifferentialActionModelNumDiff*>(bp::args("self", "model"),
                                                "Create numerical differentiation diff-action data.\n\n"
                                                ":param model: numdiff diff-action model"))
      .add_property(
          "Rx", bp::make_getter(&DifferentialActionDataNumDiff::Rx, bp::return_value_policy<bp::return_by_value>()),
          "Jacobian of the cost residual.")
      .add_property(
          "Ru", bp::make_getter(&DifferentialActionDataNumDiff::Ru, bp::return_value_policy<bp::return_by_value>()),
          "Jacobian of the cost residual.")
      .add_property(
          "dx", bp::make_getter(&DifferentialActionDataNumDiff::dx, bp::return_value_policy<bp::return_by_value>()),
          "state disturbance.")
      .add_property(
          "du", bp::make_getter(&DifferentialActionDataNumDiff::du, bp::return_value_policy<bp::return_by_value>()),
          "control disturbance.")
      .add_property(
          "xp", bp::make_getter(&DifferentialActionDataNumDiff::du, bp::return_value_policy<bp::return_by_value>()),
          "rate state after disturbance.")
      .add_property(
          "data_0",
          bp::make_getter(&DifferentialActionDataNumDiff::data_0, bp::return_value_policy<bp::return_by_value>()),
          "data that contains the final results")
      .add_property(
          "data_x",
          bp::make_getter(&DifferentialActionDataNumDiff::data_x, bp::return_value_policy<bp::return_by_value>()),
          "temporary data associated with the state variation")
      .add_property(
          "data_u",
          bp::make_getter(&DifferentialActionDataNumDiff::data_u, bp::return_value_policy<bp::return_by_value>()),
          "temporary data associated with the control variation");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
