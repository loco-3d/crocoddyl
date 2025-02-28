///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, CTU, INRIA,
//                          Heriot-Watt University, University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "diff_action.hpp"

#include "contact.hpp"
#include "cost.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/residuals/joint-acceleration.hpp"
#include "crocoddyl/core/residuals/joint-effort.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<DifferentialActionModelTypes::Type>
    DifferentialActionModelTypes::all(DifferentialActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         DifferentialActionModelTypes::Type type) {
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      os << "DifferentialActionModelLQR";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      os << "DifferentialActionModelLQRDriftFree";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelRandomLQR:
      os << "DifferentialActionModelRandomLQR";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeFwdDynamics_Hector:
      os << "DifferentialActionModelFreeFwdDynamics_Hector";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeFwdDynamics_TalosArm:
      os << "DifferentialActionModelFreeFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed:
      os << "DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeInvDynamics_Hector:
      os << "DifferentialActionModelFreeInvDynamics_Hector";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeInvDynamics_TalosArm:
      os << "DifferentialActionModelFreeInvDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeInvDynamics_TalosArm_Squashed:
      os << "DifferentialActionModelFreeInvDynamics_TalosArm_Squashed";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamics_TalosArm:
      os << "DifferentialActionModelContactFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact2DFwdDynamics_TalosArm:
      os << "DifferentialActionModelContact2DFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamics_HyQ:
      os << "DifferentialActionModelContactFwdDynamics_HyQ";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamics_Talos:
      os << "DifferentialActionModelContactFwdDynamics_Talos";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact2DFwdDynamicsWithFriction_TalosArm:
      os << "DifferentialActionModelContact2DFwdDynamicsWithFriction_TalosArm";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamicsWithFriction_HyQ:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_HyQ";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamicsWithFriction_Talos:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_Talos";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamics_TalosArm:
      os << "DifferentialActionModelContactInvDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamics_HyQ:
      os << "DifferentialActionModelContactInvDynamics_HyQ";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamics_Talos:
      os << "DifferentialActionModelContactInvDynamics_Talos";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamicsWithFriction_TalosArm:
      os << "DifferentialActionModelContactInvDynamicsWithFriction_TalosArm";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamicsWithFriction_HyQ:
      os << "DifferentialActionModelContactInvDynamicsWithFriction_HyQ";
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamicsWithFriction_Talos:
      os << "DifferentialActionModelContactInvDynamicsWithFriction_Talos";
      break;
    case DifferentialActionModelTypes::NbDifferentialActionModelTypes:
      os << "NbDifferentialActionModelTypes";
      break;
    default:
      break;
  }
  return os;
}

DifferentialActionModelFactory::DifferentialActionModelFactory() {}
DifferentialActionModelFactory::~DifferentialActionModelFactory() {}

std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>
DifferentialActionModelFactory::create(DifferentialActionModelTypes::Type type,
                                       bool with_baumgarte) const {
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> action;
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      action = std::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40,
                                                                       false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      action =
          std::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40, true);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelRandomLQR:
      action = std::make_shared<crocoddyl::DifferentialActionModelLQR>(
          crocoddyl::DifferentialActionModelLQR::Random(40, 40));
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeFwdDynamics_Hector:
      action = create_freeFwdDynamics(
          StateModelTypes::StateMultibody_Hector,
          ActuationModelTypes::ActuationModelFloatingBaseThrusters, false);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeFwdDynamics_TalosArm:
      action = create_freeFwdDynamics(StateModelTypes::StateMultibody_TalosArm,
                                      ActuationModelTypes::ActuationModelFull);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed:
      action = create_freeFwdDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelSquashingFull);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeInvDynamics_Hector:
      action = create_freeInvDynamics(
          StateModelTypes::StateMultibody_Hector,
          ActuationModelTypes::ActuationModelFloatingBaseThrusters);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeInvDynamics_TalosArm:
      action = create_freeInvDynamics(StateModelTypes::StateMultibody_TalosArm,
                                      ActuationModelTypes::ActuationModelFull);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeInvDynamics_TalosArm_Squashed:
      action = create_freeInvDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelSquashingFull);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamics_TalosArm:
      action = create_contactFwdDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelFull, false, with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact2DFwdDynamics_TalosArm:
      action = create_contactFwdDynamics(
          StateModelTypes::StateMultibodyContact2D_TalosArm,
          ActuationModelTypes::ActuationModelFull, false, with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamics_HyQ:
      action = create_contactFwdDynamics(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamics_Talos:
      action = create_contactFwdDynamics(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm:
      action = create_contactFwdDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelFull, true, with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact2DFwdDynamicsWithFriction_TalosArm:
      action = create_contactFwdDynamics(
          StateModelTypes::StateMultibodyContact2D_TalosArm,
          ActuationModelTypes::ActuationModelFull, true, with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamicsWithFriction_HyQ:
      action = create_contactFwdDynamics(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, true,
          with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactFwdDynamicsWithFriction_Talos:
      action = create_contactFwdDynamics(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, true,
          with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamics_TalosArm:
      action = create_contactInvDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamics_HyQ:
      action = create_contactInvDynamics(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamics_Talos:
      action = create_contactInvDynamics(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, false,
          with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamicsWithFriction_TalosArm:
      action = create_contactInvDynamics(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelFull, true, with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamicsWithFriction_HyQ:
      action = create_contactInvDynamics(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, true,
          with_baumgarte);
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContactInvDynamicsWithFriction_Talos:
      action = create_contactInvDynamics(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, true,
          with_baumgarte);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong DifferentialActionModelTypes::Type given");
      break;
  }
  return action;
}

std::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics>
DifferentialActionModelFactory::create_freeFwdDynamics(
    StateModelTypes::Type state_type, ActuationModelTypes::Type actuation_type,
    bool constraints) const {
  std::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> action;
  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  std::shared_ptr<crocoddyl::CostModelSum> cost;
  std::shared_ptr<crocoddyl::ConstraintModelManager> constraint;
  state = std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  const std::size_t nu = actuation->get_nu();
  cost = std::make_shared<crocoddyl::CostModelSum>(state, nu);
  cost->addCost("state",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualState, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                1.);
  cost->addCost("control",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualControl, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                1.);
  cost->addCost(
      "joint_eff",
      std::make_shared<crocoddyl::CostModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelJointEffort>(
                     state, actuation, Eigen::VectorXd::Zero(nu), nu, true)),
      1.);
  cost->addCost(
      "joint_acc",
      std::make_shared<crocoddyl::CostModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelJointAcceleration>(
                     state, nu)),
      0.01);
  cost->addCost("frame",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualFramePlacement, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                1.);
  if (constraints) {
    constraint = std::make_shared<crocoddyl::ConstraintModelManager>(state, nu);
    constraint->addConstraint(
        "frame",
        ConstraintModelFactory().create(
            ConstraintModelTypes::ConstraintModelResidualFramePlacementEquality,
            state_type, nu));
    constraint->addConstraint(
        "frame-velocity",
        ConstraintModelFactory().create(
            ConstraintModelTypes::ConstraintModelResidualFrameVelocityEquality,
            state_type, nu));
    action =
        std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
            state, actuation, cost, constraint);
  } else {
    action =
        std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
            state, actuation, cost);
  }
  return action;
}

std::shared_ptr<crocoddyl::DifferentialActionModelFreeInvDynamics>
DifferentialActionModelFactory::create_freeInvDynamics(
    StateModelTypes::Type state_type, ActuationModelTypes::Type actuation_type,
    bool constraints) const {
  std::shared_ptr<crocoddyl::DifferentialActionModelFreeInvDynamics> action;
  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  std::shared_ptr<crocoddyl::CostModelSum> cost;
  std::shared_ptr<crocoddyl::ConstraintModelManager> constraint;
  state = std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  const std::size_t nu = state->get_nv();
  cost = std::make_shared<crocoddyl::CostModelSum>(state, nu);
  cost->addCost("state",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualState, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                1.);
  cost->addCost("control",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualControl, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                1.);
  cost->addCost("frame",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualFramePlacement, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                1.);
  if (constraints) {
    constraint = std::make_shared<crocoddyl::ConstraintModelManager>(state, nu);
    constraint->addConstraint(
        "frame",
        ConstraintModelFactory().create(
            ConstraintModelTypes::ConstraintModelResidualFramePlacementEquality,
            state_type, nu));
    constraint->addConstraint(
        "frame-velocity",
        ConstraintModelFactory().create(
            ConstraintModelTypes::ConstraintModelResidualFrameVelocityEquality,
            state_type, nu));
    action =
        std::make_shared<crocoddyl::DifferentialActionModelFreeInvDynamics>(
            state, actuation, cost, constraint);
  } else {
    action =
        std::make_shared<crocoddyl::DifferentialActionModelFreeInvDynamics>(
            state, actuation, cost);
  }
  return action;
}

std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
DifferentialActionModelFactory::create_contactFwdDynamics(
    StateModelTypes::Type state_type, ActuationModelTypes::Type actuation_type,
    bool with_friction, bool with_baumgarte) const {
  std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> action;
  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  std::shared_ptr<crocoddyl::ContactModelMultiple> contact;
  std::shared_ptr<crocoddyl::CostModelSum> cost;
  state = std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  const std::size_t nu = actuation->get_nu();
  contact = std::make_shared<crocoddyl::ContactModelMultiple>(state, nu);
  cost = std::make_shared<crocoddyl::CostModelSum>(state, nu);

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  pinocchio::Force force = pinocchio::Force::Zero();
  crocoddyl::FrictionCone friction_cone(R, 0.8, 4, false);
  crocoddyl::WrenchCone wrench_cone(R, 0.8, Eigen::Vector2d(0.1, 0.1), 4,
                                    false);
  crocoddyl::ActivationBounds friction_bounds(friction_cone.get_lb(),
                                              friction_cone.get_ub());
  crocoddyl::ActivationBounds wrench_bounds(wrench_cone.get_lb(),
                                            wrench_cone.get_ub());
  std::shared_ptr<crocoddyl::ActivationModelAbstract> friction_activation =
      std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(
          friction_bounds);
  std::shared_ptr<crocoddyl::ActivationModelAbstract> wrench_activation =
      std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(
          wrench_bounds);
  Eigen::Vector2d gains = Eigen::Vector2d::Random();
  if (!with_baumgarte) {
    gains.setZero();
  }

  switch (state_type) {
    case StateModelTypes::StateMultibody_TalosArm:
      contact->addContact("lf", ContactModelFactory().create(
                                    ContactModelTypes::ContactModel3D_LOCAL,
                                    PinocchioModelTypes::TalosArm, gains,
                                    "gripper_left_fingertip_1_link", nu));
      if (with_friction) {
        // friction cone
        cost->addCost(
            "lf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state,
                    state->get_pinocchio()->getFrameId(
                        "gripper_left_fingertip_1_link"),
                    friction_cone, nu)),
            0.1);
        // force regularization
        cost->addCost(
            "lf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state,
                           state->get_pinocchio()->getFrameId(
                               "gripper_left_fingertip_1_link"),
                           force, 3, nu)),
            0.1);
      }
      break;
    case StateModelTypes::StateMultibodyContact2D_TalosArm:
      contact->addContact("lf", ContactModelFactory().create(
                                    ContactModelTypes::ContactModel2D,
                                    PinocchioModelTypes::TalosArm, gains,
                                    "gripper_left_fingertip_1_link", nu));
      if (with_friction) {
        // friction cone
        cost->addCost(
            "lf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state,
                    state->get_pinocchio()->getFrameId(
                        "gripper_left_fingertip_1_link"),
                    friction_cone, nu)),
            0.1);
        // TODO: enable force regularization once it would support Contact2D
      }
      break;
    case StateModelTypes::StateMultibody_HyQ:
      contact->addContact(
          "lf", ContactModelFactory().create(
                    ContactModelTypes::ContactModel3D_LOCAL,
                    PinocchioModelTypes::HyQ, 0.25 * gains, "lf_foot", nu));
      contact->addContact(
          "rf", ContactModelFactory().create(
                    ContactModelTypes::ContactModel3D_WORLD,
                    PinocchioModelTypes::HyQ, 0.25 * gains, "rf_foot", nu));
      contact->addContact(
          "lh", ContactModelFactory().create(
                    ContactModelTypes::ContactModel3D_LWA,
                    PinocchioModelTypes::HyQ, 0.25 * gains, "lh_foot", nu));
      contact->addContact(
          "rh", ContactModelFactory().create(
                    ContactModelTypes::ContactModel3D_LOCAL,
                    PinocchioModelTypes::HyQ, 0.25 * gains, "rh_foot", nu));
      if (with_friction) {
        // friction cone
        cost->addCost(
            "lf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("lf_foot"),
                    friction_cone, nu)),
            0.1);
        cost->addCost(
            "rf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("rf_foot"),
                    friction_cone, nu)),
            0.1);
        cost->addCost(
            "lh_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("lh_foot"),
                    friction_cone, nu)),
            0.1);
        cost->addCost(
            "rh_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("rh_foot"),
                    friction_cone, nu)),
            0.1);
        // force regularization
        cost->addCost(
            "lf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("lf_foot"),
                           force, 3, nu)),
            0.1);
        cost->addCost(
            "rf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("rf_foot"),
                           force, 3, nu)),
            0.1);
        cost->addCost(
            "lh_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("lh_foot"),
                           force, 3, nu)),
            0.1);
        cost->addCost(
            "rh_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("rh_foot"),
                           force, 3, nu)),
            0.1);
      }
      break;
    case StateModelTypes::StateMultibody_Talos:
      contact->addContact(
          "lf", ContactModelFactory().create(
                    ContactModelTypes::ContactModel6D_LOCAL,
                    PinocchioModelTypes::Talos, gains, "left_sole_link", nu));
      contact->addContact(
          "rf", ContactModelFactory().create(
                    ContactModelTypes::ContactModel6D_WORLD,
                    PinocchioModelTypes::Talos, gains, "right_sole_link", nu));
      if (with_friction) {
        // friction / wrench cone
        cost->addCost(
            "lf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("left_sole_link"),
                    friction_cone, nu)),
            0.01);
        cost->addCost(
            "rf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, wrench_activation,
                std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                    state,
                    state->get_pinocchio()->getFrameId("right_sole_link"),
                    wrench_cone, nu)),
            0.01);
        // force regularization
        cost->addCost(
            "lf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state,
                std::make_shared<crocoddyl::ResidualModelContactForce>(
                    state, state->get_pinocchio()->getFrameId("left_sole_link"),
                    force, 6, nu)),
            0.01);
        cost->addCost(
            "rf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state,
                std::make_shared<crocoddyl::ResidualModelContactForce>(
                    state,
                    state->get_pinocchio()->getFrameId("right_sole_link"),
                    force, 6, nu)),
            0.01);
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  cost->addCost("state",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualState, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                0.1);
  cost->addCost("control",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualControl, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                0.1);
  cost->addCost(
      "joint_eff",
      std::make_shared<crocoddyl::CostModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelJointEffort>(
                     state, actuation, Eigen::VectorXd::Zero(nu), nu, true)),
      0.1);
  action =
      std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(
          state, actuation, contact, cost, 0., true);
  return action;
}

std::shared_ptr<crocoddyl::DifferentialActionModelContactInvDynamics>
DifferentialActionModelFactory::create_contactInvDynamics(
    StateModelTypes::Type state_type, ActuationModelTypes::Type actuation_type,
    bool with_friction, bool with_baumgarte) const {
  std::shared_ptr<crocoddyl::DifferentialActionModelContactInvDynamics> action;
  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  std::shared_ptr<crocoddyl::ContactModelMultiple> contact;
  std::shared_ptr<crocoddyl::CostModelSum> cost;
  state = std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  std::size_t nu = state->get_nv();

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  pinocchio::Force force = pinocchio::Force::Zero();
  crocoddyl::FrictionCone friction_cone(R, 0.8, 4, false);
  crocoddyl::WrenchCone wrench_cone(R, 0.8, Eigen::Vector2d(0.1, 0.1), 4,
                                    false);
  crocoddyl::ActivationBounds friction_bounds(friction_cone.get_lb(),
                                              friction_cone.get_ub());
  crocoddyl::ActivationBounds wrench_bounds(wrench_cone.get_lb(),
                                            wrench_cone.get_ub());
  std::shared_ptr<crocoddyl::ActivationModelAbstract> friction_activation =
      std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(
          friction_bounds);
  std::shared_ptr<crocoddyl::ActivationModelAbstract> wrench_activation =
      std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(
          wrench_bounds);
  Eigen::Vector2d gains = Eigen::Vector2d::Random();
  if (!with_baumgarte) {
    gains.setZero();
  }

  switch (state_type) {
    case StateModelTypes::StateMultibody_TalosArm:
      nu += 3;
      contact = std::make_shared<crocoddyl::ContactModelMultiple>(state, nu);
      cost = std::make_shared<crocoddyl::CostModelSum>(state, nu);
      contact->addContact("lf", ContactModelFactory().create(
                                    ContactModelTypes::ContactModel3D_LOCAL,
                                    PinocchioModelTypes::TalosArm, gains,
                                    "gripper_left_fingertip_1_link", nu));
      if (with_friction) {
        // friction cone
        cost->addCost(
            "lf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state,
                    state->get_pinocchio()->getFrameId(
                        "gripper_left_fingertip_1_link"),
                    friction_cone, nu, false)),
            0.1);
        // force regularization
        cost->addCost(
            "lf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state,
                           state->get_pinocchio()->getFrameId(
                               "gripper_left_fingertip_1_link"),
                           force, 3, nu, false)),
            0.1);
      }
      break;
    case StateModelTypes::StateMultibody_HyQ:
      nu += 12;  // it includes nc
      contact = std::make_shared<crocoddyl::ContactModelMultiple>(state, nu);
      cost = std::make_shared<crocoddyl::CostModelSum>(state, nu);
      contact->addContact("lf",
                          ContactModelFactory().create(
                              ContactModelTypes::ContactModel3D_LOCAL,
                              PinocchioModelTypes::HyQ, gains, "lf_foot", nu));
      contact->addContact("rf",
                          ContactModelFactory().create(
                              ContactModelTypes::ContactModel3D_WORLD,
                              PinocchioModelTypes::HyQ, gains, "rf_foot", nu));
      contact->addContact("lh",
                          ContactModelFactory().create(
                              ContactModelTypes::ContactModel3D_LWA,
                              PinocchioModelTypes::HyQ, gains, "lh_foot", nu));
      contact->addContact("rh",
                          ContactModelFactory().create(
                              ContactModelTypes::ContactModel3D_LOCAL,
                              PinocchioModelTypes::HyQ, gains, "rh_foot", nu));
      if (with_friction) {
        // friction cone
        cost->addCost(
            "lf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("lf_foot"),
                    friction_cone, nu, false)),
            0.1);
        cost->addCost(
            "rf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("rf_foot"),
                    friction_cone, nu, false)),
            0.1);
        cost->addCost(
            "lh_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("lh_foot"),
                    friction_cone, nu, false)),
            0.1);
        cost->addCost(
            "rh_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("rh_foot"),
                    friction_cone, nu, false)),
            0.1);
        // force regularization
        cost->addCost(
            "lf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("lf_foot"),
                           force, 3, nu, false)),
            0.1);
        cost->addCost(
            "rf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("rf_foot"),
                           force, 3, nu, false)),
            0.1);
        cost->addCost(
            "lh_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("lh_foot"),
                           force, 3, nu, false)),
            0.1);
        cost->addCost(
            "rh_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("rh_foot"),
                           force, 3, nu, false)),
            0.1);
      }
      break;
    case StateModelTypes::StateMultibody_Talos:
      nu += 12;  // it includes nc
      contact = std::make_shared<crocoddyl::ContactModelMultiple>(state, nu);
      cost = std::make_shared<crocoddyl::CostModelSum>(state, nu);
      contact->addContact(
          "lf", ContactModelFactory().create(
                    ContactModelTypes::ContactModel6D_LOCAL,
                    PinocchioModelTypes::Talos, gains, "left_sole_link", nu));
      contact->addContact(
          "rf", ContactModelFactory().create(
                    ContactModelTypes::ContactModel6D_WORLD,
                    PinocchioModelTypes::Talos, gains, "right_sole_link", nu));
      if (with_friction) {
        // friction / wrench cone
        cost->addCost(
            "lf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("left_sole_link"),
                    friction_cone, nu, false)),
            0.01);
        cost->addCost(
            "rf_cone",
            std::make_shared<crocoddyl::CostModelResidual>(
                state, wrench_activation,
                std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                    state,
                    state->get_pinocchio()->getFrameId("right_sole_link"),
                    wrench_cone, nu, false)),
            0.01);
        // force regularization
        cost->addCost(
            "lf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state,
                std::make_shared<crocoddyl::ResidualModelContactForce>(
                    state, state->get_pinocchio()->getFrameId("left_sole_link"),
                    force, 6, nu, false)),
            0.01);
        cost->addCost(
            "rf_forceReg",
            std::make_shared<crocoddyl::CostModelResidual>(
                state,
                std::make_shared<crocoddyl::ResidualModelContactForce>(
                    state,
                    state->get_pinocchio()->getFrameId("right_sole_link"),
                    force, 6, nu, false)),
            0.01);
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  cost->addCost("state",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualState, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                0.1);
  cost->addCost("control",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualControl, state_type,
                    ActivationModelTypes::ActivationModelQuad, nu),
                0.1);
  action =
      std::make_shared<crocoddyl::DifferentialActionModelContactInvDynamics>(
          state, actuation, contact, cost);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
