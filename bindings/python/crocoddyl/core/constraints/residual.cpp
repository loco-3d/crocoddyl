///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2024, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/constraints/residual.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeConstraintResidual() {
  bp::register_ptr_to_python<std::shared_ptr<ConstraintModelResidual> >();

  bp::class_<ConstraintModelResidual, bp::bases<ConstraintModelAbstract> >(
      "ConstraintModelResidual",
      "This defines equality / inequality constraints based on a residual "
      "vector and its bounds.",
      bp::init<std::shared_ptr<StateAbstract>,
               std::shared_ptr<ResidualModelAbstract>, Eigen::VectorXd,
               Eigen::VectorXd, bp::optional<bool> >(
          bp::args("self", "state", "residual", "lower", "upper", "T_act"),
          "Initialize the residual constraint model as an inequality "
          "constraint.\n\n"
          ":param state: state description\n"
          ":param residual: residual model\n"
          ":param lower: lower bound\n"
          ":param upper: upper bound\n"
          ":param T_act: false if we want to deactivate the residual at the "
          "terminal node (default true)"))
      .def(bp::init<std::shared_ptr<StateAbstract>,
                    std::shared_ptr<ResidualModelAbstract>,
                    bp::optional<bool> >(
          bp::args("self", "state", "residual", "T_act"),
          "Initialize the residual constraint model as an equality "
          "constraint.\n\n"
          ":param state: state description\n"
          ":param residual: residual model\n"
          ":param T_act: false if we want to deactivate the residual at the "
          "terminal node (default true)"))
      .def<void (ConstraintModelResidual::*)(
          const std::shared_ptr<ConstraintDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelResidual::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the residual constraint.\n\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ConstraintModelResidual::*)(
          const std::shared_ptr<ConstraintDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelAbstract::calc, bp::args("self", "data", "x"),
          "Compute the residual constraint based on state only.\n\n"
          "It updates the constraint based on the state only.\n"
          "This function is commonly used in the terminal nodes of an optimal "
          "control problem.\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)")
      .def<void (ConstraintModelResidual::*)(
          const std::shared_ptr<ConstraintDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelResidual::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the residual constraint.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)\n")
      .def<void (ConstraintModelResidual::*)(
          const std::shared_ptr<ConstraintDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelAbstract::calcDiff,
          bp::args("self", "data", "x"),
          "Compute the derivatives of the residual constraint with respect to "
          "the state only.\n\n"
          "It updates the Jacobian of the constraint function based on the "
          "state only.\n"
          "This function is commonly used in the terminal nodes of an optimal "
          "control problem.\n"
          ":param data: constraint data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &ConstraintModelResidual::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the residual constraint data.\n\n"
           "Each constraint model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for a predefined constraint.\n"
           ":param data: shared data\n"
           ":return constraint data.")
      .def(CopyableVisitor<ConstraintModelResidual>());

  bp::register_ptr_to_python<std::shared_ptr<ConstraintDataResidual> >();

  bp::class_<ConstraintDataResidual, bp::bases<ConstraintDataAbstract> >(
      "ConstraintDataResidual", "Data for residual constraint.\n\n",
      bp::init<ConstraintModelResidual*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create residual constraint data.\n\n"
          ":param model: residual constraint model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .def(CopyableVisitor<ConstraintDataResidual>());
}

}  // namespace python
}  // namespace crocoddyl
