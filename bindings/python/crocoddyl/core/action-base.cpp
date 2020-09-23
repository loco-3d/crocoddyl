///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/utils/printable.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {
namespace python {

void exposeActionAbstract() {
  // Register custom converters between std::vector and Python list
  typedef boost::shared_ptr<ActionModelAbstract> ActionModelPtr;
  typedef boost::shared_ptr<ActionDataAbstract> ActionDataPtr;
  StdVectorPythonVisitor<ActionModelPtr, std::allocator<ActionModelPtr>, true>::expose("StdVec_ActionModel");
  StdVectorPythonVisitor<ActionDataPtr, std::allocator<ActionDataPtr>, true>::expose("StdVec_ActionData");

  bp::register_ptr_to_python<boost::shared_ptr<ActionModelAbstract> >();

  bp::class_<ActionModelAbstract_wrap, boost::noncopyable>(
      "ActionModelAbstract",
      "Abstract class for action models.\n\n"
      "An action model combines dynamics and cost data. Each node, in our optimal control\n"
      "problem, is described through an action model. Every time that we want to describe\n"
      "a problem, we need to provide ways of computing the dynamics, cost functions and their\n"
      "derivatives. These computations are mainly carrying on inside calc() and calcDiff(),\n"
      "respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, std::size_t, bp::optional<std::size_t, std::size_t, std::size_t> >(
          bp::args("self", "state", "nu", "nr", "ng"),
          "Initialize the action model.\n\n"
          "You can also describe autonomous systems by setting nu = 0.\n"
          ":param state: state description,\n"
          ":param nu: dimension of control vector,\n"
          ":param nr: dimension of the cost-residual vector (default 1)\n"
          ":param ng: number of inequality constraints (default 0)\n"
          ":param nh: number of equality constraints (default 0)\n"))
      .def("calc", pure_virtual(&ActionModelAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the next state and cost value.\n\n"
           "It describes the time-discrete evolution of our dynamical system\n"
           "in which we obtain the next state. Additionally it computes the\n"
           "cost value associated to this discrete state and control pair.\n"
           ":param data: action data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def<void (ActionModelAbstract::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"),
          "Compute the total cost value for nodes that depends only on the state.\n\n"
          "It updates the total cost and the next state is not computed as it is not expected to change.\n"
          "This function is used in the terminal nodes of an optimal control problem.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)")
      .def("calcDiff", pure_virtual(&ActionModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the dynamics and cost functions.\n\n"
           "It computes the partial derivatives of the dynamical system and the\n"
           "cost function. It assumes that calc has been run first.\n"
           "This function builds a quadratic approximation of the\n"
           "action model (i.e. linear dynamics and quadratic cost).\n"
           ":param data: action data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def<void (ActionModelAbstract::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff, bp::args("self", "data", "x"),
          "Compute the derivatives of the cost functions with respect to the state only.\n\n"
          "It updates the derivatives of the cost function with respect to the state only.\n"
          "This function is used in the terminal nodes of an optimal control problem.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &ActionModelAbstract_wrap::createData, &ActionModelAbstract_wrap::default_createData,
           bp::args("self"),
           "Create the action data.\n\n"
           "Each action model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .def("quasiStatic", &ActionModelAbstract_wrap::quasiStatic_x,
           ActionModel_quasiStatic_wraps(
               bp::args("self", "data", "x", "maxiter", "tol"),
               "Compute the quasic-static control given a state.\n\n"
               "It runs an iterative Newton step in order to compute the quasic-static regime\n"
               "given a state configuration.\n"
               ":param data: action data\n"
               ":param x: discrete-time state vector\n"
               ":param maxiter: maximum allowed number of iterations\n"
               ":param tol: stopping tolerance criteria (default 1e-9)\n"
               ":return u: quasic-static control"))
      .def("quasiStatic", &ActionModelAbstract_wrap::quasiStatic, &ActionModelAbstract_wrap::default_quasiStatic,
           bp::args("self", "data", "u", "x", "maxiter", "tol"))
      .def("multiplyByFx", &ActionModelAbstract_wrap::multiplyByFx, &ActionModelAbstract_wrap::default_multiplyByFx,
           bp::args("self", "Fx", "A", "out", "op"),
           "Compute the product between the given matrix A and the Jacobian of the dynamics with respect to the "
           "state.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fx: Jacobian matrix of the dynamics with respect to the state\n"
           ":param A: matrix to multiply (dim na x state.ndx)\n"
           ":param out: product between A and the Jacobian of the dynamics with respect to the state (dim na x "
           "state.ndx)\n"
           ":param op: assignment operator which sets, adds, or removes the given results")
      .def("multiplyByFx", &ActionModelAbstract_wrap::multiplyByFx_A, bp::args("self", "Fx", "A"),
           "Compute the product between the given matrix A and the Jacobian of the dynamics with respect to the "
           "state.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fx: Jacobian matrix of the dynamics with respect to the state\n"
           ":param A: matrix to multiply (dim na x state.ndx)\n"
           "return out: product between A and the Jacobian of the dynamics with respect to the state (dim na x "
           "state.ndx)")
      .def("multiplyFxTransposeBy", &ActionModelAbstract_wrap::multiplyFxTransposeBy,
           &ActionModelAbstract_wrap::default_multiplyFxTransposeBy, bp::args("self", "Fx", "A", "out", "op"),
           "Compute the product between the transpose of the Jacobian of the dynamics with respect to the state\n"
           "and a given matrix A.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fx: Jacobian matrix of the dynamics with respect to the state\n"
           ":param A: matrix to multiply (dim state.ndx x na)\n"
           ":param out: product between the transpose of the Jacobian of the dynamics with respec the state and A "
           "(dim state.ndx x na)\n"
           ":param op: assignment operator which sets, adds, or removes the given results")
      .def("multiplyFxTransposeBy", &ActionModelAbstract_wrap::multiplyFxTransposeBy_A, bp::args("self", "Fx", "A"),
           "Compute the product between the transpose of the Jacobian of the dynamics with respect to the state\n"
           "and a given matrix A.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fx: Jacobian matrix of the dynamics with respect to the state\n"
           ":param A: matrix to multiply (dim state.ndx x na)\n"
           ":return product between the transpose of the Jacobian of the dynamics with respec the state and A (dim "
           "state.ndx x na)")
      .def("multiplyByFu", &ActionModelAbstract_wrap::multiplyByFu, &ActionModelAbstract_wrap::default_multiplyByFu,
           bp::args("self", "Fu", "A", "out", "op"),
           "Compute the product between the given matrix A and the Jacobian of the dynamics with respect to the "
           "control.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fu: Jacobian matrix of the dynamics with respect to the control\n"
           ":param A: matrix to multiply (dim na x state.ndx)\n"
           ":param out: product between A and the Jacobian of the dynamics with respect to the control (dim na x nu)\n"
           ":param op: assignment operator which sets, adds, or removes the given results")
      .def("multiplyByFu", &ActionModelAbstract_wrap::multiplyByFu_A, bp::args("self", "Fu", "A"),
           "Compute the product between the given matrix A and the Jacobian of the dynamics with respect to the "
           "control.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fu: Jacobian matrix of the dynamics with respect to the control\n"
           ":param A: matrix to multiply (dim na x state.ndx)\n"
           "return out: product between A and the Jacobian of the dynamics with respect to the control (dim na x nu)")
      .def("multiplyFuTransposeBy", &ActionModelAbstract_wrap::multiplyFuTransposeBy,
           &ActionModelAbstract_wrap::default_multiplyFuTransposeBy, bp::args("self", "Fu", "A", "out", "op"),
           "Compute the product between the transpose of the Jacobian of the dynamics with respect to the control\n"
           "and a given matrix A.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fu: Jacobian matrix of the dynamics with respect to the control\n"
           ":param A: matrix to multiply (dim state.ndx x na)\n"
           ":param out: product between the transpose of the Jacobian of the dynamics with respec the control and A "
           "(dim nu x na)\n"
           ":param op: assignment operator which sets, adds, or removes the given results")
      .def("multiplyFuTransposeBy", &ActionModelAbstract_wrap::multiplyFuTransposeBy_A, bp::args("self", "Fu", "A"),
           "Compute the product between the transpose of the Jacobian of the dynamics with respect to the control\n"
           "and a given matrix A.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fu: Jacobian matrix of the dynamics with respect to the control\n"
           ":param A: matrix to multiply (dim state.ndx x na)\n"
           ":return product between the transpose of the Jacobian of the dynamics with respec the control and A (dim "
           "nu x na)")
      .add_property("nu", bp::make_function(&ActionModelAbstract_wrap::get_nu), "dimension of control vector")
      .add_property("nr", bp::make_function(&ActionModelAbstract_wrap::get_nr), "dimension of cost-residual vector")
      .add_property("ng", bp::make_function(&ActionModelAbstract_wrap::get_ng), "number of inequality constraints")
      .add_property("nh", bp::make_function(&ActionModelAbstract_wrap::get_nh), "number of equality constraints")
      .add_property(
          "state",
          bp::make_function(&ActionModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state")
      .add_property("has_control_limits", bp::make_function(&ActionModelAbstract_wrap::get_has_control_limits),
                    "indicates whether problem has finite control limits")
      .add_property("u_lb", bp::make_function(&ActionModelAbstract_wrap::get_u_lb, bp::return_internal_reference<>()),
                    &ActionModelAbstract_wrap::set_u_lb, "lower control limits")
      .add_property("u_ub", bp::make_function(&ActionModelAbstract_wrap::get_u_ub, bp::return_internal_reference<>()),
                    &ActionModelAbstract_wrap::set_u_ub, "upper control limits")
      .def(PrintableVisitor<ActionModelAbstract>());

  bp::register_ptr_to_python<boost::shared_ptr<ActionDataAbstract> >();

  bp::class_<ActionDataAbstract, boost::noncopyable>(
      "ActionDataAbstract",
      "Abstract class for action data.\n\n"
      "In crocoddyl, an action data contains all the required information for processing an\n"
      "user-defined action model. The action data typically is allocated onces by running\n"
      "model.createData() and contains the first- and second- order derivatives of the dynamics\n"
      "and cost function, respectively.",
      bp::init<ActionModelAbstract*>(bp::args("self", "model"),
                                     "Create common data shared between AMs.\n\n"
                                     "The action data uses the model in order to first process it.\n"
                                     ":param model: action model"))
      .add_property("cost", bp::make_getter(&ActionDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::cost), "cost value")
      .add_property("xnext", bp::make_getter(&ActionDataAbstract::xnext, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::xnext), "next state")
      .add_property("r", bp::make_getter(&ActionDataAbstract::r, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::r), "cost residual")
      .add_property("Fx", bp::make_getter(&ActionDataAbstract::Fx, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Fx), "Jacobian of the dynamics")
      .add_property("Fu", bp::make_getter(&ActionDataAbstract::Fu, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Fu), "Jacobian of the dynamics")
      .add_property("Lx", bp::make_getter(&ActionDataAbstract::Lx, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Lx), "Jacobian of the cost")
      .add_property("Lu", bp::make_getter(&ActionDataAbstract::Lu, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Lu), "Jacobian of the cost")
      .add_property("Lxx", bp::make_getter(&ActionDataAbstract::Lxx, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Lxx), "Hessian of the cost")
      .add_property("Lxu", bp::make_getter(&ActionDataAbstract::Lxu, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Lxu), "Hessian of the cost")
      .add_property("Luu", bp::make_getter(&ActionDataAbstract::Luu, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Luu), "Hessian of the cost")
      .add_property("g", bp::make_getter(&ActionDataAbstract::g, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::g), "Inequality constraint values")
      .add_property("Gx", bp::make_getter(&ActionDataAbstract::Gx, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Gx), "Jacobian of the inequality constraint")
      .add_property("Gu", bp::make_getter(&ActionDataAbstract::Gu, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Gu), "Jacobian of the inequality constraint")
      .add_property("h", bp::make_getter(&ActionDataAbstract::h, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::h), "Equality constraint values")
      .add_property("Hx", bp::make_getter(&ActionDataAbstract::Hx, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Hx), "Jacobian of the equality constraint")
      .add_property("Hu", bp::make_getter(&ActionDataAbstract::Hu, bp::return_internal_reference<>()),
                    bp::make_setter(&ActionDataAbstract::Hu), "Jacobian of the equality constraint");
}

}  // namespace python
}  // namespace crocoddyl
