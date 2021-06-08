///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Oxford,
//                     University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/integr-action-base.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionModelAbstract> >();

  bp::class_<IntegratedActionModelAbstract_wrap, boost::noncopyable>(
      "IntegratedActionModelAbstract",
      "Abstract class for integrated action models.\n\n"
      "In crocoddyl, an integrated action model trasforms a differential action model in a (discrete) action model.\n",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, double, bool >(
          bp::args("self", "model", "timestep", "with_cost_residual"),
          "Initialize the action model.\n\n"
          "You can also describe autonomous systems by setting nu = 0.\n"
          ":param model: differential action model,\n"
          ":param timestep: integration time step,\n"
          ":param with_cost_residual: bool flag"))
      .def("calc", pure_virtual(&IntegratedActionModelAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the next state and cost value.\n\n"
           "It describes the time-discrete evolution of our dynamical system\n"
           "in which we obtain the next state. Additionally it computes the\n"
           "cost value associated to this discrete state and control pair.\n"
           ":param data: action data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input")
      .def<void (IntegratedActionModelAbstract::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &IntegratedActionModelAbstract::calc,
                                                                                    bp::args("self", "data", "x"))
      .def("calcDiff", pure_virtual(&IntegratedActionModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the dynamics and cost functions.\n\n"
           "It computes the partial derivatives of the dynamical system and the\n"
           "cost function. It assumes that calc has been run first.\n"
           "This function builds a quadratic approximation of the\n"
           "action model (i.e. linear dynamics and quadratic cost).\n"
           ":param data: action data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input\n")
      .def<void (IntegratedActionModelAbstract::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelAbstract_wrap::createData, &IntegratedActionModelAbstract_wrap::default_createData,
           bp::args("self"),
           "Create the action data.\n\n"
           "Each action model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property(
          "nu", bp::make_function(&IntegratedActionModelAbstract_wrap::get_nu, bp::return_value_policy<bp::return_by_value>()),
          "dimension of control vector")
      .add_property(
          "nr", bp::make_function(&IntegratedActionModelAbstract_wrap::get_nr, bp::return_value_policy<bp::return_by_value>()),
          "dimension of cost-residual vector")
      .add_property(
          "nu_diff", bp::make_function(&IntegratedActionModelAbstract_wrap::get_nu_diff, bp::return_value_policy<bp::return_by_value>()),
          "dimension of control vector of the differentiable model")
      .add_property(
          "state",
          bp::make_function(&IntegratedActionModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state")
      .add_property("has_control_limits", bp::make_function(&IntegratedActionModelAbstract_wrap::get_has_control_limits),
                    "indicates whether problem has finite control limits")
      .add_property("u_lb", bp::make_function(&IntegratedActionModelAbstract_wrap::get_u_lb, bp::return_internal_reference<>()),
                    &IntegratedActionModelAbstract_wrap::set_u_lb, "lower control limits")
      .add_property("u_ub", bp::make_function(&IntegratedActionModelAbstract_wrap::get_u_ub, bp::return_internal_reference<>()),
                    &IntegratedActionModelAbstract_wrap::set_u_ub, "upper control limits")
      .def(PrintableVisitor<IntegratedActionModelAbstract>());

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataAbstract> >();

  bp::class_<IntegratedActionDataAbstract, bp::bases<ActionDataAbstract> >(
      "IntegratedActionDataAbstract",
      "Abstract class for integrated action data.\n\n"
      "In crocoddyl, an action data contains all the required information for processing an\n"
      "user-defined action model. The action data typically is allocated onces by running\n"
      "model.createData() and contains the first- and second- order derivatives of the dynamics\n"
      "and cost function, respectively.",
      bp::init<IntegratedActionModelAbstract*>(bp::args("self", "model"),
                                     "Create common data shared between AMs.\n\n"
                                     "The action data uses the model in order to first process it.\n"
                                     ":param model: action model"))
      .add_property("cost", bp::make_getter(&IntegratedActionDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&IntegratedActionDataAbstract::cost), "cost value")
      .add_property("xnext", bp::make_getter(&IntegratedActionDataAbstract::xnext, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::xnext), "next state")
      .add_property("r", bp::make_getter(&IntegratedActionDataAbstract::r, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::r), "cost residual")
      .add_property("Fx", bp::make_getter(&IntegratedActionDataAbstract::Fx, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::Fx), "Jacobian of the dynamics")
      .add_property("Fu", bp::make_getter(&IntegratedActionDataAbstract::Fu, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::Fu), "Jacobian of the dynamics")
      .add_property("Lx", bp::make_getter(&IntegratedActionDataAbstract::Lx, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::Lx), "Jacobian of the cost")
      .add_property("Lu", bp::make_getter(&IntegratedActionDataAbstract::Lu, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::Lu), "Jacobian of the cost")
      .add_property("Lxx", bp::make_getter(&IntegratedActionDataAbstract::Lxx, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::Lxx), "Hessian of the cost")
      .add_property("Lxu", bp::make_getter(&IntegratedActionDataAbstract::Lxu, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::Lxu), "Hessian of the cost")
      .add_property("Luu", bp::make_getter(&IntegratedActionDataAbstract::Luu, bp::return_internal_reference<>()),
                    bp::make_setter(&IntegratedActionDataAbstract::Luu), "Hessian of the cost");
}

}  // namespace python
}  // namespace crocoddyl
