///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actions/impulse-fwddyn.hpp"

#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeActionImpulseFwdDynamics() {
  bp::register_ptr_to_python<std::shared_ptr<ActionModelImpulseFwdDynamics> >();

  bp::class_<ActionModelImpulseFwdDynamics, bp::bases<ActionModelAbstract> >(
      "ActionModelImpulseFwdDynamics",
      "Action model for impulse forward dynamics in multibody systems.\n\n"
      "The impulse is modelled as holonomic constraits in the contact frame. "
      "There\n"
      "is also a custom implementation in case of system with armatures. If "
      "you want to\n"
      "include the armature, you need to use set_armature(). On the other "
      "hand, the\n"
      "stack of cost functions are implemented in CostModelSum().",
      bp::init<std::shared_ptr<StateMultibody>,
               std::shared_ptr<ImpulseModelMultiple>,
               std::shared_ptr<CostModelSum>,
               bp::optional<double, double, bool> >(
          bp::args("self", "state", " impulses", "costs", "r_coeff",
                   "inv_damping", "enable_force"),
          "Initialize the impulse forward-dynamics action model.\n\n"
          "The damping factor is needed when the contact Jacobian is not "
          "full-rank. Otherwise,\n"
          "a good damping factor could be 1e-12. In addition, if you have cost "
          "based on forces,\n"
          "you need to enable the computation of the force Jacobians (i.e. "
          "enable_force=True)."
          ":param state: multibody state\n"
          ":param impulses: multiple impulse model\n"
          ":param costs: stack of cost functions\n"
          ":param r_coeff: restitution coefficient (default 0.)\n"
          ":param inv_damping: Damping factor for cholesky decomposition of "
          "JMinvJt (default 0.)\n"
          ":param enable_force: Enable the computation of force Jacobians "
          "(default False)"))
      .def(bp::init<std::shared_ptr<StateMultibody>,
                    std::shared_ptr<ImpulseModelMultiple>,
                    std::shared_ptr<CostModelSum>,
                    std::shared_ptr<ConstraintModelManager>,
                    bp::optional<double, double, bool> >(
          bp::args("self", "state", "impulses", "costs", "constraints",
                   "r_coeff", "inv_damping", "enable_force"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          "The damping factor is needed when the contact Jacobian is not "
          "full-rank. Otherwise,\n"
          "a good damping factor could be 1e-12. In addition, if you have cost "
          "based on forces,\n"
          "you need to enable the computation of the force Jacobians (i.e. "
          "enable_force=True)."
          ":param state: multibody state\n"
          ":param impulses: multiple impulse model\n"
          ":param costs: stack of cost functions\n"
          ":param constraints: stack of constraint functions\n"
          ":param r_coeff: restitution coefficient (default 0.)\n"
          ":param inv_damping: Damping factor for cholesky decomposition of "
          "JMinvJt (default 0.)\n"
          ":param enable_force: Enable the computation of force Jacobians "
          "(default False)"))
      .def<void (ActionModelImpulseFwdDynamics::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelImpulseFwdDynamics::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-continuous evolution of the multibody system "
          "with impulse. The\n"
          "impulses are modelled as holonomic constraints.\n"
          "Additionally it computes the cost value associated to this state "
          "and control pair.\n"
          ":param data: impulse forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input")
      .def<void (ActionModelImpulseFwdDynamics::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ActionModelImpulseFwdDynamics::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelImpulseFwdDynamics::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the differential multibody system and "
          "its cost\n"
          "functions.\n\n"
          "It computes the partial derivatives of the differential multibody "
          "system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: impulse forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input\n"
          "")
      .def<void (ActionModelImpulseFwdDynamics::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ActionModelImpulseFwdDynamics::createData,
           bp::args("self"),
           "Create the impulse forward dynamics differential action data.")
      .add_property(
          "pinocchio",
          bp::make_function(&ActionModelImpulseFwdDynamics::get_pinocchio,
                            bp::return_internal_reference<>()),
          "multibody model (i.e. pinocchio model)")
      .add_property(
          "impulses",
          bp::make_function(&ActionModelImpulseFwdDynamics::get_impulses,
                            bp::return_value_policy<bp::return_by_value>()),
          "multiple contact model")
      .add_property(
          "costs",
          bp::make_function(&ActionModelImpulseFwdDynamics::get_costs,
                            bp::return_value_policy<bp::return_by_value>()),
          "total cost model")
      .add_property(
          "constraints",
          bp::make_function(&ActionModelImpulseFwdDynamics::get_constraints,
                            bp::return_value_policy<bp::return_by_value>()),
          "constraint model manager")
      .add_property(
          "armature",
          bp::make_function(&ActionModelImpulseFwdDynamics::get_armature,
                            bp::return_internal_reference<>()),
          bp::make_function(&ActionModelImpulseFwdDynamics::set_armature),
          "set an armature mechanism in the joints")
      .add_property(
          "r_coeff",
          bp::make_function(
              &ActionModelImpulseFwdDynamics::get_restitution_coefficient),
          bp::make_function(
              &ActionModelImpulseFwdDynamics::set_restitution_coefficient),
          "Restitution coefficient that describes elastic impacts")
      .add_property(
          "JMinvJt_damping",
          bp::make_function(&ActionModelImpulseFwdDynamics::get_damping_factor),
          bp::make_function(&ActionModelImpulseFwdDynamics::set_damping_factor),
          "Damping factor for cholesky decomposition of JMinvJt")
      .def(CopyableVisitor<ActionModelImpulseFwdDynamics>());

  bp::register_ptr_to_python<std::shared_ptr<ActionDataImpulseFwdDynamics> >();

  bp::class_<ActionDataImpulseFwdDynamics, bp::bases<ActionDataAbstract> >(
      "ActionDataImpulseFwdDynamics",
      "Action data for the impulse forward dynamics system.",
      bp::init<ActionModelImpulseFwdDynamics*>(
          bp::args("self", "model"),
          "Create impulse forward-dynamics action data.\n\n"
          ":param model: impulse forward-dynamics action model"))
      .add_property("pinocchio",
                    bp::make_getter(&ActionDataImpulseFwdDynamics::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("multibody",
                    bp::make_getter(&ActionDataImpulseFwdDynamics::multibody,
                                    bp::return_internal_reference<>()),
                    "multibody data")
      .add_property(
          "costs",
          bp::make_getter(&ActionDataImpulseFwdDynamics::costs,
                          bp::return_value_policy<bp::return_by_value>()),
          "total cost data")
      .add_property(
          "constraints",
          bp::make_getter(&ActionDataImpulseFwdDynamics::constraints,
                          bp::return_value_policy<bp::return_by_value>()),
          "constraint data")
      .add_property("Kinv",
                    bp::make_getter(&ActionDataImpulseFwdDynamics::Kinv,
                                    bp::return_internal_reference<>()),
                    "inverse of the KKT matrix")
      .add_property("df_dx",
                    bp::make_getter(&ActionDataImpulseFwdDynamics::df_dx,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the contact impulse")
      .def(CopyableVisitor<ActionDataImpulseFwdDynamics>());
}

}  // namespace python
}  // namespace crocoddyl
