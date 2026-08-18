///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "dynamics.hpp"

#include "constraint.hpp"
#include "cost.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/residuals/joint-acceleration.hpp"
#include "crocoddyl/core/residuals/joint-effort.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/dynamics/constrained-inverse.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"

namespace crocoddyl {
namespace unittest {

namespace {

struct ContactDescription {
  std::string name;
  pinocchio::FrameIndex frame;
  pinocchio::ReferenceFrame reference_frame;
  std::size_t nc;
};

ContactModel::MaskArray contact_mask(const std::size_t nc) {
  ContactModel::MaskArray mask = {{false, false, false, false, false, false}};
  for (std::size_t i = 0; i < nc; ++i) {
    mask[i] = true;
  }
  return mask;
}

std::size_t contact_dimension(const StateModelTypes::Type state_type) {
  switch (state_type) {
    case StateModelTypes::StateMultibody_TalosArm:
      return 3;
    case StateModelTypes::StateMultibody_HyQ:
    case StateModelTypes::StateMultibody_Talos:
      return 12;
    default:
      throw_pretty(__FILE__ ": Unsupported contact state type");
  }
}

std::vector<ContactDescription> add_contacts(
    const StateModelTypes::Type state_type,
    const std::shared_ptr<StateMultibody>& state,
    const std::shared_ptr<ImplicitConstraintModelMultiple>& constraints,
    const Eigen::Vector2d& gains) {
  std::vector<ContactDescription> contacts;
  switch (state_type) {
    case StateModelTypes::StateMultibody_TalosArm:
      contacts.push_back(ContactDescription{
          "lf",
          state->get_pinocchio()->getFrameId("gripper_left_fingertip_1_link"),
          pinocchio::LOCAL, 3});
      break;
    case StateModelTypes::StateMultibody_HyQ:
      contacts.push_back(ContactDescription{
          "lf", state->get_pinocchio()->getFrameId("lf_foot"), pinocchio::LOCAL,
          3});
      contacts.push_back(ContactDescription{
          "rf", state->get_pinocchio()->getFrameId("rf_foot"), pinocchio::WORLD,
          3});
      contacts.push_back(ContactDescription{
          "lh", state->get_pinocchio()->getFrameId("lh_foot"),
          pinocchio::LOCAL_WORLD_ALIGNED, 3});
      contacts.push_back(ContactDescription{
          "rh", state->get_pinocchio()->getFrameId("rh_foot"), pinocchio::LOCAL,
          3});
      break;
    case StateModelTypes::StateMultibody_Talos:
      contacts.push_back(ContactDescription{
          "lf", state->get_pinocchio()->getFrameId("left_sole_link"),
          pinocchio::LOCAL, 6});
      contacts.push_back(ContactDescription{
          "rf", state->get_pinocchio()->getFrameId("right_sole_link"),
          pinocchio::WORLD, 6});
      break;
    default:
      throw_pretty(__FILE__ ": Unsupported contact state type");
  }

  for (std::vector<ContactDescription>::const_iterator it = contacts.begin();
       it != contacts.end(); ++it) {
    constraints->addConstraint(
        it->name,
        std::make_shared<ContactModel>(
            state, it->frame, pinocchio::SE3::Identity(), it->reference_frame,
            constraints->get_nu(), gains, contact_mask(it->nc)));
  }
  return contacts;
}

void add_standard_costs(const std::shared_ptr<CostModelSum>& costs,
                        const StateModelTypes::Type state_type,
                        const std::size_t nu, const double weight) {
  costs->addCost("state",
                 CostModelFactory().create(
                     CostModelTypes::CostModelResidualState, state_type,
                     ActivationModelTypes::ActivationModelQuad, nu),
                 weight);
  costs->addCost("control",
                 CostModelFactory().create(
                     CostModelTypes::CostModelResidualControl, state_type,
                     ActivationModelTypes::ActivationModelQuad, nu),
                 weight);
}

void add_frame_cost(const std::shared_ptr<CostModelSum>& costs,
                    const StateModelTypes::Type state_type,
                    const std::size_t nu, const double weight) {
  costs->addCost("frame",
                 CostModelFactory().create(
                     CostModelTypes::CostModelResidualFramePlacement,
                     state_type, ActivationModelTypes::ActivationModelQuad, nu),
                 weight);
}

std::shared_ptr<ConstraintModelManager> create_action_constraints(
    const std::shared_ptr<StateMultibody>& state,
    const StateModelTypes::Type state_type, const std::size_t nu) {
  const std::shared_ptr<ConstraintModelManager> constraints =
      std::make_shared<ConstraintModelManager>(state, nu);
  constraints->addConstraint(
      "frame",
      ConstraintModelFactory().create(
          ConstraintModelTypes::ConstraintModelResidualFramePlacementEquality,
          state_type, nu));
  constraints->addConstraint(
      "frame-velocity",
      ConstraintModelFactory().create(
          ConstraintModelTypes::ConstraintModelResidualFrameVelocityEquality,
          state_type, nu));
  return constraints;
}

void add_contact_costs(const std::shared_ptr<CostModelSum>& costs,
                       const std::shared_ptr<StateMultibody>& state,
                       const std::vector<ContactDescription>& contacts,
                       const std::size_t nu, const bool fwddyn) {
  const Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  const FrictionCone friction_cone(rotation, 0.8, 4, false);
  const WrenchCone wrench_cone(rotation, 0.8, Eigen::Vector2d(0.1, 0.1), 4,
                               false);
  const std::shared_ptr<ActivationModelAbstract> friction_activation =
      std::make_shared<ActivationModelQuadraticBarrier>(
          ActivationBounds(friction_cone.get_lb(), friction_cone.get_ub()));
  const std::shared_ptr<ActivationModelAbstract> wrench_activation =
      std::make_shared<ActivationModelQuadraticBarrier>(
          ActivationBounds(wrench_cone.get_lb(), wrench_cone.get_ub()));

  for (std::vector<ContactDescription>::const_iterator it = contacts.begin();
       it != contacts.end(); ++it) {
    if (it->nc == 6 && it->name == "rf") {
      costs->addCost(it->name + "_cone",
                     std::make_shared<CostModelResidual>(
                         state, wrench_activation,
                         std::make_shared<ResidualModelContactWrenchCone>(
                             state, it->frame, wrench_cone, nu, fwddyn)),
                     0.01);
    } else {
      costs->addCost(it->name + "_cone",
                     std::make_shared<CostModelResidual>(
                         state, friction_activation,
                         std::make_shared<ResidualModelContactFrictionCone>(
                             state, it->frame, friction_cone, nu, fwddyn)),
                     it->nc == 6 ? 0.01 : 0.1);
    }
    costs->addCost(it->name + "_force",
                   std::make_shared<CostModelResidual>(
                       state, std::make_shared<ResidualModelContactForce>(
                                  state, it->frame, pinocchio::Force::Zero(),
                                  it->nc, nu, fwddyn)),
                   it->nc == 6 ? 0.01 : 0.1);
  }
}

}  // namespace

const std::vector<DynamicsModelTypes::Type> DynamicsModelTypes::all(
    DynamicsModelTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         const DynamicsModelTypes::Type type) {
  switch (type) {
    case DynamicsModelTypes::DynamicsModelFreeFwd_Hector:
      os << "DynamicsModelFreeFwd_Hector";
      break;
    case DynamicsModelTypes::DynamicsModelFreeFwd_TalosArm:
      os << "DynamicsModelFreeFwd_TalosArm";
      break;
    case DynamicsModelTypes::DynamicsModelFreeInv_Hector:
      os << "DynamicsModelFreeInv_Hector";
      break;
    case DynamicsModelTypes::DynamicsModelFreeInv_TalosArm:
      os << "DynamicsModelFreeInv_TalosArm";
      break;
    case DynamicsModelTypes::DynamicsModelContactFwd_TalosArm:
      os << "DynamicsModelContactFwd_TalosArm";
      break;
    case DynamicsModelTypes::DynamicsModelContactFwd_HyQ:
      os << "DynamicsModelContactFwd_HyQ";
      break;
    case DynamicsModelTypes::DynamicsModelContactFwd_Talos:
      os << "DynamicsModelContactFwd_Talos";
      break;
    case DynamicsModelTypes::DynamicsModelContactFwdWithFriction_HyQ:
      os << "DynamicsModelContactFwdWithFriction_HyQ";
      break;
    case DynamicsModelTypes::DynamicsModelContactFwdWithFriction_Talos:
      os << "DynamicsModelContactFwdWithFriction_Talos";
      break;
    case DynamicsModelTypes::DynamicsModelContactInv_TalosArm:
      os << "DynamicsModelContactInv_TalosArm";
      break;
    case DynamicsModelTypes::DynamicsModelContactInv_HyQ:
      os << "DynamicsModelContactInv_HyQ";
      break;
    case DynamicsModelTypes::DynamicsModelContactInv_Talos:
      os << "DynamicsModelContactInv_Talos";
      break;
    case DynamicsModelTypes::DynamicsModelContactInvWithFriction_HyQ:
      os << "DynamicsModelContactInvWithFriction_HyQ";
      break;
    case DynamicsModelTypes::DynamicsModelContactInvWithFriction_Talos:
      os << "DynamicsModelContactInvWithFriction_Talos";
      break;
    case DynamicsModelTypes::NbDynamicsModelTypes:
      os << "NbDynamicsModelTypes";
      break;
    default:
      break;
  }
  return os;
}

DynamicsModelFactory::DynamicsModelFactory() {}
DynamicsModelFactory::~DynamicsModelFactory() {}

DynamicsModelFactoryResult DynamicsModelFactory::create(
    const DynamicsModelTypes::Type type, const bool with_baumgarte) const {
  switch (type) {
    case DynamicsModelTypes::DynamicsModelFreeFwd_Hector:
      return create_freeFwdDynamics(
          StateModelTypes::StateMultibody_Hector,
          ActuationModelTypes::ActuationModelFloatingBaseThrusters, false);
    case DynamicsModelTypes::DynamicsModelFreeFwd_TalosArm:
      return create_freeFwdDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelMultibody);
    case DynamicsModelTypes::DynamicsModelFreeInv_Hector:
      return create_freeInvDynamics(
          StateModelTypes::StateMultibody_Hector,
          ActuationModelTypes::ActuationModelFloatingBaseThrusters);
    case DynamicsModelTypes::DynamicsModelFreeInv_TalosArm:
      return create_freeInvDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelMultibody);
    case DynamicsModelTypes::DynamicsModelContactFwd_TalosArm:
      return create_contactFwdDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelMultibody, false, with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactFwd_HyQ:
      return create_contactFwdDynamics(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactFwd_Talos:
      return create_contactFwdDynamics(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactFwdWithFriction_HyQ:
      return create_contactFwdDynamics(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, true,
          with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactFwdWithFriction_Talos:
      return create_contactFwdDynamics(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, true,
          with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactInv_TalosArm:
      return create_contactInvDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactInv_HyQ:
      return create_contactInvDynamics(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactInv_Talos:
      return create_contactInvDynamics(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactInvWithFriction_HyQ:
      return create_contactInvDynamics(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, true,
          with_baumgarte);
    case DynamicsModelTypes::DynamicsModelContactInvWithFriction_Talos:
      return create_contactInvDynamics(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, true,
          with_baumgarte);
    default:
      throw_pretty(__FILE__ ": Wrong DynamicsModelTypes::Type given");
  }
}

DynamicsModelFactoryResult DynamicsModelFactory::create_freeFwdDynamics(
    const StateModelTypes::Type state_type,
    const ActuationModelTypes::Type actuation_type,
    const bool with_action_constraints) const {
  DynamicsModelFactoryResult result;
  const std::shared_ptr<StateMultibody> state =
      std::static_pointer_cast<StateMultibody>(
          StateModelFactory().create(state_type));
  const std::shared_ptr<ActuationModelAbstract> actuation =
      ActuationModelFactory().create(actuation_type, state_type);
  const std::size_t nu = actuation->get_nu();
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state, nu);
  result.dynamics = std::make_shared<DynamicsModelConstrainedForward>(
      state, actuation, constraints);
  result.costs = std::make_shared<CostModelSum>(state, nu);
  add_standard_costs(result.costs, state_type, nu, 1.);
  result.costs->addCost(
      "joint-effort",
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelJointEffort>(
                     state, actuation,
                     Eigen::VectorXd::Zero(actuation->get_nu()), nu, true)),
      1.);
  result.costs->addCost(
      "joint-acceleration",
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelJointAcceleration>(state, nu)),
      0.01);
  add_frame_cost(result.costs, state_type, nu, 1.);
  if (with_action_constraints) {
    result.constraints = create_action_constraints(state, state_type, nu);
  }
  return result;
}

DynamicsModelFactoryResult DynamicsModelFactory::create_freeInvDynamics(
    const StateModelTypes::Type state_type,
    const ActuationModelTypes::Type actuation_type,
    const bool with_action_constraints) const {
  DynamicsModelFactoryResult result;
  const std::shared_ptr<StateMultibody> state =
      std::static_pointer_cast<StateMultibody>(
          StateModelFactory().create(state_type));
  const std::shared_ptr<ActuationModelAbstract> actuation =
      ActuationModelFactory().create(actuation_type, state_type);
  const std::size_t nu = state->get_nv();
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state, nu);
  result.dynamics = std::make_shared<DynamicsModelConstrainedInverse>(
      state, actuation, constraints);
  result.costs = std::make_shared<CostModelSum>(state, nu);
  add_standard_costs(result.costs, state_type, nu, 1.);
  add_frame_cost(result.costs, state_type, nu, 1.);
  if (with_action_constraints) {
    result.constraints = create_action_constraints(state, state_type, nu);
  }
  return result;
}

DynamicsModelFactoryResult DynamicsModelFactory::create_contactFwdDynamics(
    const StateModelTypes::Type state_type,
    const ActuationModelTypes::Type actuation_type, const bool with_friction,
    const bool with_baumgarte) const {
  DynamicsModelFactoryResult result;
  const std::shared_ptr<StateMultibody> state =
      std::static_pointer_cast<StateMultibody>(
          StateModelFactory().create(state_type));
  const std::shared_ptr<ActuationModelAbstract> actuation =
      ActuationModelFactory().create(actuation_type, state_type);
  const std::size_t nu = actuation->get_nu();
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state, nu);
  const Eigen::Vector2d gains =
      with_baumgarte ? Eigen::Vector2d(0.2, 0.1) : Eigen::Vector2d::Zero();
  const std::vector<ContactDescription> contacts =
      add_contacts(state_type, state, constraints, gains);
  result.dynamics = std::make_shared<DynamicsModelConstrainedForward>(
      state, actuation, constraints);
  result.costs = std::make_shared<CostModelSum>(state, nu);
  add_standard_costs(result.costs, state_type, nu, 0.1);
  result.costs->addCost(
      "joint-effort",
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelJointEffort>(
                     state, actuation,
                     Eigen::VectorXd::Zero(actuation->get_nu()), nu, true)),
      0.1);
  if (with_friction) {
    add_contact_costs(result.costs, state, contacts, nu, true);
  }
  return result;
}

DynamicsModelFactoryResult DynamicsModelFactory::create_contactInvDynamics(
    const StateModelTypes::Type state_type,
    const ActuationModelTypes::Type actuation_type, const bool with_friction,
    const bool with_baumgarte) const {
  DynamicsModelFactoryResult result;
  const std::shared_ptr<StateMultibody> state =
      std::static_pointer_cast<StateMultibody>(
          StateModelFactory().create(state_type));
  const std::shared_ptr<ActuationModelAbstract> actuation =
      ActuationModelFactory().create(actuation_type, state_type);
  const std::size_t nu = state->get_nv() + contact_dimension(state_type);
  const std::shared_ptr<ImplicitConstraintModelMultiple> constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state, nu);
  const Eigen::Vector2d gains =
      with_baumgarte ? Eigen::Vector2d(0.2, 0.1) : Eigen::Vector2d::Zero();
  const std::vector<ContactDescription> contacts =
      add_contacts(state_type, state, constraints, gains);
  result.dynamics = std::make_shared<DynamicsModelConstrainedInverse>(
      state, actuation, constraints);
  result.costs = std::make_shared<CostModelSum>(state, nu);
  add_standard_costs(result.costs, state_type, nu, 0.1);
  if (with_friction) {
    add_contact_costs(result.costs, state, contacts, nu, false);
  }
  return result;
}

}  // namespace unittest
}  // namespace crocoddyl
