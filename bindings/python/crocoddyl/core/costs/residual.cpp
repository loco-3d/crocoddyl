///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/costs/residual.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeCostResidual() {
  bp::register_ptr_to_python<std::shared_ptr<CostModelResidual> >();

  bp::class_<CostModelResidual, bp::bases<CostModelAbstract> >(
      "CostModelResidual",
      "This cost function uses a residual vector with a Gauss-Newton "
      "assumption to define a cost term.",
      bp::init<std::shared_ptr<StateAbstract>,
               std::shared_ptr<ActivationModelAbstract>,
               std::shared_ptr<ResidualModelAbstract> >(
          bp::args("self", "state", "activation", "residual"),
          "Initialize the residual cost model.\n\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param residual: residual model"))
      .def(bp::init<std::shared_ptr<StateAbstract>,
                    std::shared_ptr<ResidualModelAbstract> >(
          bp::args("self", "state", "residual"),
          "Initialize the residual cost model.\n\n"
          ":param state: state description\n"
          ":param residual: residual model"))
      .def<void (CostModelResidual::*)(
          const std::shared_ptr<CostDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelResidual::calc, bp::args("self", "data", "x", "u"),
          "Compute the residual cost.\n\n"
          ":param data: cost residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (CostModelResidual::*)(
          const std::shared_ptr<CostDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"),
          "Compute the residual cost based on state only.\n\n"
          "It updates the total cost based on the state only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: cost data\n"
          ":param x: state point (dim. state.nx)")
      .def<void (CostModelResidual::*)(
          const std::shared_ptr<CostDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelResidual::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the residual cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: cost residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (CostModelResidual::*)(
          const std::shared_ptr<CostDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff,
          bp::args("self", "data", "x"),
          "Compute the derivatives of the residual cost with respect to the "
          "state only.\n\n"
          "It updates the Jacobian and Hessian of the cost function based on "
          "the state only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: cost residual data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &CostModelResidual::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the residual cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This "
           "function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .def(CopyableVisitor<CostModelResidual>());

  bp::register_ptr_to_python<std::shared_ptr<CostDataResidual> >();

  bp::class_<CostDataResidual, bp::bases<CostDataAbstract> >(
      "CostDataResidual", "Data for residual cost.\n\n",
      bp::init<CostModelResidual*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create residual cost data.\n\n"
          ":param model: residual cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .def(CopyableVisitor<CostDataResidual>());
}

}  // namespace python
}  // namespace crocoddyl
