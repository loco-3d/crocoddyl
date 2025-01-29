///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/diff-action.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeDifferentialActionNumDiff() {
  bp::register_ptr_to_python<
      std::shared_ptr<DifferentialActionModelNumDiff> >();

  bp::class_<DifferentialActionModelNumDiff,
             bp::bases<DifferentialActionModelAbstract> >(
      "DifferentialActionModelNumDiff",
      "Abstract class for computing calcDiff by using numerical "
      "differentiation.\n\n",
      bp::init<std::shared_ptr<DifferentialActionModelAbstract>,
               bp::optional<bool> >(
          bp::args("self", "model", "gaussApprox"),
          "Initialize the action model NumDiff.\n\n"
          ":param model: action model where we compute the derivatives through "
          "NumDiff,\n"
          ":param gaussApprox: compute the Hessian using Gauss approximation "
          "(default False)"))
      .def<void (DifferentialActionModelNumDiff::*)(
          const std::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DifferentialActionModelNumDiff::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the next state and cost value.\n\n"
          "The system evolution is described in model.\n"
          ":param data: NumDiff action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (DifferentialActionModelNumDiff::*)(
          const std::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DifferentialActionModelAbstract::calc,
          bp::args("self", "data", "x"))
      .def<void (DifferentialActionModelNumDiff::*)(
          const std::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DifferentialActionModelNumDiff::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the dynamics and cost functions.\n\n"
          "It computes the Jacobian and Hessian using numerical "
          "differentiation.\n"
          "It assumes that calc has been run first.\n"
          ":param data: NumDiff action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (DifferentialActionModelNumDiff::*)(
          const std::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DifferentialActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &DifferentialActionModelNumDiff::createData,
           bp::args("self"),
           "Create the action data.\n\n"
           "Each action model (AM) has its own data that needs to be "
           "allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property(
          "model",
          bp::make_function(&DifferentialActionModelNumDiff::get_model,
                            bp::return_value_policy<bp::return_by_value>()),
          "action model")
      .add_property(
          "disturbance",
          bp::make_function(&DifferentialActionModelNumDiff::get_disturbance),
          &DifferentialActionModelNumDiff::set_disturbance,
          "disturbance constant used in the numerical differentiation")
      .add_property("withGaussApprox",
                    bp::make_function(
                        &DifferentialActionModelNumDiff::get_with_gauss_approx),
                    "Gauss approximation for computing the Hessians")
      .def(CopyableVisitor<DifferentialActionModelNumDiff>());

  bp::register_ptr_to_python<std::shared_ptr<DifferentialActionDataNumDiff> >();

  bp::class_<DifferentialActionDataNumDiff,
             bp::bases<DifferentialActionDataAbstract> >(
      "DifferentialActionDataNumDiff",
      "Numerical differentiation diff-action data.",
      bp::init<DifferentialActionModelNumDiff*>(
          bp::args("self", "model"),
          "Create numerical differentiation diff-action data.\n\n"
          ":param model: numdiff diff-action model"))
      .add_property("Rx",
                    bp::make_getter(&DifferentialActionDataNumDiff::Rx,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the cost residual.")
      .add_property("Ru",
                    bp::make_getter(&DifferentialActionDataNumDiff::Ru,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the cost residual.")
      .add_property("dx",
                    bp::make_getter(&DifferentialActionDataNumDiff::dx,
                                    bp::return_internal_reference<>()),
                    "state disturbance.")
      .add_property("du",
                    bp::make_getter(&DifferentialActionDataNumDiff::du,
                                    bp::return_internal_reference<>()),
                    "control disturbance.")
      .add_property("xp",
                    bp::make_getter(&DifferentialActionDataNumDiff::xp,
                                    bp::return_internal_reference<>()),
                    "rate state after disturbance.")
      .add_property(
          "data_0",
          bp::make_getter(&DifferentialActionDataNumDiff::data_0,
                          bp::return_value_policy<bp::return_by_value>()),
          "data that contains the final results")
      .add_property(
          "data_x",
          bp::make_getter(&DifferentialActionDataNumDiff::data_x,
                          bp::return_value_policy<bp::return_by_value>()),
          "temporary data associated with the state variation")
      .add_property(
          "data_u",
          bp::make_getter(&DifferentialActionDataNumDiff::data_u,
                          bp::return_value_policy<bp::return_by_value>()),
          "temporary data associated with the control variation")
      .def(CopyableVisitor<DifferentialActionDataNumDiff>());
}

}  // namespace python
}  // namespace crocoddyl
