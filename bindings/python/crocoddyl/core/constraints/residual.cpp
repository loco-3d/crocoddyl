///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/constraints/residual.hpp"

namespace crocoddyl {
namespace python {

void exposeConstraintResidual() {
  bp::register_ptr_to_python<boost::shared_ptr<ConstraintModelResidual> >();

  bp::class_<ConstraintModelResidual, bp::bases<ConstraintModelAbstract> >(
      "ConstraintModelResidual",
      "This defines equality / inequality constraints based on a residual vector and its bounds.",
      bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ResidualModelAbstract>, Eigen::VectorXd,
               Eigen::VectorXd>(bp::args("self", "state", "residual", "lower", "upper"),
                                "Initialize the residual constraint model given its bounds.\n\n"
                                ":param state: state description\n"
                                ":param residual: residual model\n"
                                ":param lower: lower bound\n"
                                ":param upper: upper bound"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ResidualModelAbstract> >(
          bp::args("self", "state", "residual"),
          "Initialize the residual constraint model as equality constraint equals to zero.\n\n"
          ":param state: state description\n"
          ":param residual: residual model"))
      .def<void (ConstraintModelResidual::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelResidual::calc, bp::args("self", "data", "x", "u"),
          "Compute the residual constraint.\n\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ConstraintModelResidual::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelAbstract::calc, bp::args("self", "data", "x"),
          "Compute the residual constraint based on state only.\n\n"
          "It updates the constraint based on the state only.\n"
          "This function is commonly used in the terminal nodes of an optimal control problem.\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)")
      .def<void (ConstraintModelResidual::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelResidual::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the residual constraint.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)\n")
      .def<void (ConstraintModelResidual::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelAbstract::calcDiff, bp::args("self", "data", "x"),
          "Compute the derivatives of the residual constraint with respect to the state only.\n\n"
          "It updates the Jacobian of the constraint function based on the state only.\n"
          "This function is commonly used in the terminal nodes of an optimal control problem.\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &ConstraintModelResidual::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the residual constraint data.\n\n"
           "Each constraint model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined constraint.\n"
           ":param data: shared data\n"
           ":return constraint data.")
      .def("updateBounds", &ConstraintModelResidual::update_bounds, bp::args("self", "lower", "upper"),
           "Update the lower and upper bounds.\n\n"
           ":param lower: lower bound\n"
           ":param upper: lower bound")
      .add_property("lb", bp::make_function(&ConstraintModelResidual::get_lb, bp::return_internal_reference<>()))
      .add_property("ub", bp::make_function(&ConstraintModelResidual::get_ub, bp::return_internal_reference<>()));

  bp::register_ptr_to_python<boost::shared_ptr<ConstraintDataResidual> >();

  bp::class_<ConstraintDataResidual, bp::bases<ConstraintDataAbstract> >(
      "ConstraintDataResidual", "Data for residual constraint.\n\n",
      bp::init<ConstraintModelResidual*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create residual constraint data.\n\n"
          ":param model: residual constraint model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()]);
}

}  // namespace python
}  // namespace crocoddyl
