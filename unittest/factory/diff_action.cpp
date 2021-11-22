///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "diff_action.hpp"
#include "cost.hpp"
#include "contact.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<DifferentialActionModelTypes::Type> DifferentialActionModelTypes::all(
    DifferentialActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, DifferentialActionModelTypes::Type type) {
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      os << "DifferentialActionModelLQR";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      os << "DifferentialActionModelLQRDriftFree";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm:
      os << "DifferentialActionModelFreeFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed:
      os << "DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_TalosArm:
      os << "DifferentialActionModelContactFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContact2DFwdDynamics_TalosArm:
      os << "DifferentialActionModelContact2DFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_HyQ:
      os << "DifferentialActionModelContactFwdDynamics_HyQ";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_Talos:
      os << "DifferentialActionModelContactFwdDynamics_Talos";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContact2DFwdDynamicsWithFriction_TalosArm:
      os << "DifferentialActionModelContact2DFwdDynamicsWithFriction_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_HyQ:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_HyQ";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_Talos:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_Talos";
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

boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> DifferentialActionModelFactory::create(
    DifferentialActionModelTypes::Type type) const {
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> action;
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      action = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      action = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40, true);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm:
      action =
          create_freeFwdDynamics(StateModelTypes::StateMultibody_TalosArm, ActuationModelTypes::ActuationModelFull);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed:
      action = create_freeFwdDynamics(StateModelTypes::StateMultibody_TalosArm,
                                      ActuationModelTypes::ActuationModelSquashingFull);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_TalosArm:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_TalosArm,
                                         ActuationModelTypes::ActuationModelFull, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContact2DFwdDynamics_TalosArm:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibodyContact2D_TalosArm,
                                         ActuationModelTypes::ActuationModelFull, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_HyQ:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_HyQ,
                                         ActuationModelTypes::ActuationModelFloatingBase, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_Talos:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_Talos,
                                         ActuationModelTypes::ActuationModelFloatingBase, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm:
      action =
          create_contactFwdDynamics(StateModelTypes::StateMultibody_TalosArm, ActuationModelTypes::ActuationModelFull);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContact2DFwdDynamicsWithFriction_TalosArm:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibodyContact2D_TalosArm,
                                         ActuationModelTypes::ActuationModelFull);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_HyQ:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_HyQ,
                                         ActuationModelTypes::ActuationModelFloatingBase);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_Talos:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_Talos,
                                         ActuationModelTypes::ActuationModelFloatingBase);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong DifferentialActionModelTypes::Type given");
      break;
  }
  return action;
}

boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics>
DifferentialActionModelFactory::create_freeFwdDynamics(StateModelTypes::Type state_type,
                                                       ActuationModelTypes::Type actuation_type) const {
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;
  state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  cost->addCost("state",
                CostModelFactory().create(CostModelTypes::CostModelResidualState, state_type,
                                          ActivationModelTypes::ActivationModelQuad),
                1.);
  cost->addCost("control",
                CostModelFactory().create(CostModelTypes::CostModelResidualControl, state_type,
                                          ActivationModelTypes::ActivationModelQuad),
                1.);
  cost->addCost("frame",
                CostModelFactory().create(CostModelTypes::CostModelResidualFramePlacement, state_type,
                                          ActivationModelTypes::ActivationModelQuad),
                1.);
  action = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, cost);
  return action;
}

boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
DifferentialActionModelFactory::create_contactFwdDynamics(StateModelTypes::Type state_type,
                                                          ActuationModelTypes::Type actuation_type,
                                                          bool with_friction) const {
  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contact;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;
  state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  contact = boost::make_shared<crocoddyl::ContactModelMultiple>(state, actuation->get_nu());
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  pinocchio::Force force = pinocchio::Force::Zero();
  crocoddyl::FrictionCone friction_cone(R, 0.8, 4, false);
  crocoddyl::WrenchCone wrench_cone(R, 0.8, Eigen::Vector2d(0.1, 0.1), 4, false);
  crocoddyl::ActivationBounds friction_bounds(friction_cone.get_lb(), friction_cone.get_ub());
  crocoddyl::ActivationBounds wrench_bounds(wrench_cone.get_lb(), wrench_cone.get_ub());
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> friction_activation =
      boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(friction_bounds);
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> wrench_activation =
      boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(wrench_bounds);
  switch (state_type) {
    case StateModelTypes::StateMultibody_TalosArm:
      contact->addContact("lf", boost::make_shared<crocoddyl::ContactModel3D>(
                                    state, state->get_pinocchio()->getFrameId("gripper_left_fingertip_1_link"),
                                    Eigen::Vector3d::Zero(), actuation->get_nu()));
      if (with_friction) {
        // friction cone
        cost->addCost("lf_cone",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, friction_activation,
                          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                              state, state->get_pinocchio()->getFrameId("gripper_left_fingertip_1_link"),
                              friction_cone, actuation->get_nu())),
                      0.1);
        // force regularization
        cost->addCost("lf_forceReg",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                                     state, state->get_pinocchio()->getFrameId("gripper_left_fingertip_1_link"), force,
                                     3, actuation->get_nu())),
                      0.1);
      }
      break;
    case StateModelTypes::StateMultibodyContact2D_TalosArm:
      contact->addContact("lf", boost::make_shared<crocoddyl::ContactModel2D>(
                                    state, state->get_pinocchio()->getFrameId("gripper_left_fingertip_1_link"),
                                    Eigen::Vector2d::Zero(), actuation->get_nu()));
      if (with_friction) {
        // friction cone
        cost->addCost("lf_cone",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, friction_activation,
                          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                              state, state->get_pinocchio()->getFrameId("gripper_left_fingertip_1_link"),
                              friction_cone, actuation->get_nu())),
                      0.1);
        // TODO: enable force regularization once it would support Contact2D
      }
      break;
    case StateModelTypes::StateMultibody_HyQ:
      contact->addContact(
          "lf", ContactModelFactory().create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::HyQ, "lf_foot",
                                             actuation->get_nu()));
      contact->addContact(
          "rf", ContactModelFactory().create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::HyQ, "rf_foot",
                                             actuation->get_nu()));
      contact->addContact(
          "lh", ContactModelFactory().create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::HyQ, "lh_foot",
                                             actuation->get_nu()));
      contact->addContact(
          "rh", ContactModelFactory().create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::HyQ, "rh_foot",
                                             actuation->get_nu()));
      if (with_friction) {
        // friction cone
        cost->addCost(
            "lf_cone",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("lf_foot"), friction_cone, actuation->get_nu())),
            0.1);
        cost->addCost(
            "rf_cone",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("rf_foot"), friction_cone, actuation->get_nu())),
            0.1);
        cost->addCost(
            "lh_cone",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("lh_foot"), friction_cone, actuation->get_nu())),
            0.1);
        cost->addCost(
            "rh_cone",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("rh_foot"), friction_cone, actuation->get_nu())),
            0.1);
        // force regularization
        cost->addCost(
            "lf_forceReg",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("lf_foot"), force, 3, actuation->get_nu())),
            0.1);
        cost->addCost(
            "rf_forceReg",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("rf_foot"), force, 3, actuation->get_nu())),
            0.1);
        cost->addCost(
            "lh_forceReg",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("lh_foot"), force, 3, actuation->get_nu())),
            0.1);
        cost->addCost(
            "rh_forceReg",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                           state, state->get_pinocchio()->getFrameId("rh_foot"), force, 3, actuation->get_nu())),
            0.1);
      }
      break;
    case StateModelTypes::StateMultibody_Talos:
      contact->addContact("lf",
                          ContactModelFactory().create(ContactModelTypes::ContactModel6D, PinocchioModelTypes::Talos,
                                                       "left_sole_link", actuation->get_nu()));
      contact->addContact("rf",
                          ContactModelFactory().create(ContactModelTypes::ContactModel6D, PinocchioModelTypes::Talos,
                                                       "right_sole_link", actuation->get_nu()));
      if (with_friction) {
        // friction / wrench cone
        cost->addCost(
            "lf_cone",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, friction_activation,
                boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("left_sole_link"), friction_cone, actuation->get_nu())),
            0.01);
        cost->addCost(
            "rf_cone",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, wrench_activation,
                boost::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                    state, state->get_pinocchio()->getFrameId("right_sole_link"), wrench_cone, actuation->get_nu())),
            0.01);
        // force regularization
        cost->addCost("lf_forceReg",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                                     state, state->get_pinocchio()->getFrameId("left_sole_link"), force, 6,
                                     actuation->get_nu())),
                      0.01);
        cost->addCost("rf_forceReg",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                                     state, state->get_pinocchio()->getFrameId("right_sole_link"), force, 6,
                                     actuation->get_nu())),
                      0.01);
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  cost->addCost("state",
                CostModelFactory().create(CostModelTypes::CostModelResidualState, state_type,
                                          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
                0.1);
  cost->addCost("control",
                CostModelFactory().create(CostModelTypes::CostModelResidualControl, state_type,
                                          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
                0.1);
  action = boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state, actuation, contact, cost,
                                                                                    0., true);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
