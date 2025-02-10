///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2024, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/constraint-base.hpp"

#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

void exposeConstraintAbstract() {
  bp::register_ptr_to_python<std::shared_ptr<ConstraintModelAbstract> >();

  bp::enum_<ConstraintType>("ConstraintType")
      .value("Inequality", Inequality)
      .value("Equality", Equality)
      .value("Both", Both)
      .export_values();

  bp::class_<ConstraintModelAbstract_wrap, boost::noncopyable>(
      "ConstraintModelAbstract",
      "Abstract multibody constraint models.\n\n"
      "A constraint model defines both: inequality g(x,u) and equality h(x, u) "
      "constraints.\n"
      "The constraint function depends on the state point x, which lies in the "
      "state manifold\n"
      "described with a nx-tuple, its velocity xd that belongs to the tangent "
      "space with ndx dimension,\n"
      "and the control input u.",
      bp::init<std::shared_ptr<StateAbstract>,
               std::shared_ptr<ResidualModelAbstract>, std::size_t,
               std::size_t>(bp::args("self", "state", "residual", "ng", "nh"),
                            "Initialize the constraint model.\n\n"
                            ":param state: state description\n"
                            ":param residual: residual model\n"
                            ":param ng: number of inequality constraints\n"
                            ":param nh: number of equality constraints"))
      .def(bp::init<std::shared_ptr<StateAbstract>, std::size_t, std::size_t,
                    std::size_t, bp::optional<bool> >(
          bp::args("self", "state", "nu", "ng", "nh", "T_const"),
          "Initialize the constraint model.\n\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector (default state.nv)\n"
          ":param ng: number of inequality constraints\n"
          ":param nh: number of equality constraints\n"
          ":param T_const: True if this is a constraint in both running and "
          "terminal nodes.\n"
          "                False if it is a constraint on running nodes only "
          "(default true)"))
      .def(bp::init<std::shared_ptr<StateAbstract>, std::size_t, std::size_t,
                    bp::optional<bool> >(
          bp::args("self", "state", "ng", "nh", "T_const"),
          "Initialize the constraint model.\n\n"
          ":param state: state description\n"
          ":param ng: number of inequality constraints\n"
          ":param nh: number of equality constraints\n"
          ":param T_const: True if this is a constraint in both running and "
          "terminal nodes.\n"
          "                False if it is a constraint on running nodes only "
          "(default true)"))
      .def("calc", pure_virtual(&ConstraintModelAbstract_wrap::calc),
           bp::args("self", "data", "x", "u"),
           "Compute the constraint value.\n\n"
           ":param data: constraint data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def<void (ConstraintModelAbstract::*)(
          const std::shared_ptr<ConstraintDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelAbstract::calc, bp::args("self", "data", "x"),
          "Compute the constraint value for nodes that depends only on the "
          "state.\n\n"
          "It updates the constraint based on the state only.\n"
          "This function is commonly used in the terminal nodes of an optimal "
          "control problem.\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)")
      .def("calcDiff", pure_virtual(&ConstraintModelAbstract_wrap::calcDiff),
           bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the constraint function.\n\n"
           "It computes the Jacobians of the constraint function.\n"
           "It assumes that calc has been run first.\n"
           ":param data: constraint data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)\n")
      .def<void (ConstraintModelAbstract::*)(
          const std::shared_ptr<ConstraintDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelAbstract::calcDiff,
          bp::args("self", "data", "x"),
          "Compute the Jacobian of the constraint with respect to the state "
          "only.\n\n"
          "It computes the Jacobian of the constraint function based on the "
          "state only.\n"
          "This function is commonly used in the terminal nodes of an optimal "
          "control problem.\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &ConstraintModelAbstract_wrap::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the constraint data.\n\n"
           "Each constraint model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for a predefined constraint.\n"
           ":param data: shared data\n"
           ":return constraint data.")
      .def("createData", &ConstraintModelAbstract_wrap::default_createData,
           bp::with_custodian_and_ward_postcall<0, 2>())
      .def("updateBounds", &ConstraintModelAbstract_wrap::update_bounds,
           bp::args("self", "lower", "upper"),
           "Update the lower and upper bounds.\n\n"
           ":param lower: lower bound\n"
           ":param upper: upper bound")
      .def("removeBounds", &ConstraintModelAbstract_wrap::remove_bounds,
           bp::args("self"), "Remove the bounds.")
      .add_property(
          "state",
          bp::make_function(&ConstraintModelAbstract_wrap::get_state,
                            bp::return_value_policy<bp::return_by_value>()),
          "state description")
      .add_property(
          "residual",
          bp::make_function(&ConstraintModelAbstract_wrap::get_residual,
                            bp::return_value_policy<bp::return_by_value>()),
          "residual model")
      .add_property("type",
                    bp::make_function(&ConstraintModelAbstract_wrap::get_type),
                    "type of constraint")
      .add_property("lb",
                    bp::make_function(&ConstraintModelAbstract_wrap::get_lb,
                                      bp::return_internal_reference<>()),
                    "lower bound of constraint")
      .add_property("ub",
                    bp::make_function(&ConstraintModelAbstract_wrap::get_ub,
                                      bp::return_internal_reference<>()),
                    "upper bound of constraint")
      .add_property("nu",
                    bp::make_function(&ConstraintModelAbstract_wrap::get_nu),
                    "dimension of control vector")
      .add_property("ng",
                    bp::make_function(&ConstraintModelAbstract_wrap::get_ng),
                    "number of inequality constraints")
      .add_property("nh",
                    bp::make_function(&ConstraintModelAbstract_wrap::get_nh),
                    "number of equality constraints")
      .add_property(
          "T_constraint",
          bp::make_function(&ConstraintModelAbstract_wrap::get_T_constraint),
          "True if the constraint is imposed in terminal nodes as well")
      .def(PrintableVisitor<ConstraintModelAbstract>());

  bp::register_ptr_to_python<std::shared_ptr<ConstraintDataAbstract> >();

  bp::class_<ConstraintDataAbstract>(
      "ConstraintDataAbstract", "Abstract class for constraint data.\n\n",
      bp::init<ConstraintModelAbstract*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create common data shared between constraint models.\n\n"
          ":param model: constraint model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("shared",
                    bp::make_getter(&ConstraintDataAbstract::shared,
                                    bp::return_internal_reference<>()),
                    "shared data")
      .add_property(
          "residual",
          bp::make_getter(&ConstraintDataAbstract::residual,
                          bp::return_value_policy<bp::return_by_value>()),
          "residual data")
      .add_property("g",
                    bp::make_getter(&ConstraintDataAbstract::g,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ConstraintDataAbstract::g),
                    "inequality constraint residual")
      .add_property("Gx",
                    bp::make_getter(&ConstraintDataAbstract::Gx,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ConstraintDataAbstract::Gx),
                    "Jacobian of the inequality constraint")
      .add_property("Gu",
                    bp::make_getter(&ConstraintDataAbstract::Gu,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ConstraintDataAbstract::Gu),
                    "Jacobian of the inequality constraint")
      .add_property("h",
                    bp::make_getter(&ConstraintDataAbstract::h,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ConstraintDataAbstract::h),
                    "equality constraint residual")
      .add_property("Hx",
                    bp::make_getter(&ConstraintDataAbstract::Hx,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ConstraintDataAbstract::Hx),
                    "Jacobian of the equality constraint")
      .add_property("Hu",
                    bp::make_getter(&ConstraintDataAbstract::Hu,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ConstraintDataAbstract::Hu),
                    "Jacobian of the equality constraint")
      .def(CopyableVisitor<ConstraintDataAbstract>());
}

}  // namespace python
}  // namespace crocoddyl
