///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "action.hpp"

#include "cost.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "impulse.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ActionModelTypes::Type> ActionModelTypes::all(
    ActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ActionModelTypes::Type type) {
  switch (type) {
    case ActionModelTypes::ActionModelUnicycle:
      os << "ActionModelUnicycle";
      break;
    case ActionModelTypes::ActionModelLQRDriftFree:
      os << "ActionModelLQRDriftFree";
      break;
    case ActionModelTypes::ActionModelLQR:
      os << "ActionModelLQR";
      break;
    case ActionModelTypes::ActionModelRandomLQR:
      os << "ActionModelRandomLQR";
      break;
    case ActionModelTypes::ActionModelRandomLQRwithTerminalConstraint:
      os << "ActionModelRandomLQRwithTerminalConstraint";
      break;
    case ActionModelTypes::ActionModelImpulseFwdDynamics_HyQ:
      os << "ActionModelImpulseFwdDynamics_HyQ";
      break;
    case ActionModelTypes::ActionModelImpulseFwdDynamics_Talos:
      os << "ActionModelImpulseFwdDynamics_Talos";
      break;
    case ActionModelTypes::NbActionModelTypes:
      os << "NbActionModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ActionModelFactory::ActionModelFactory() {}
ActionModelFactory::~ActionModelFactory() {}

std::shared_ptr<crocoddyl::ActionModelAbstract> ActionModelFactory::create(
    ActionModelTypes::Type type, Instance instance) const {
  std::shared_ptr<crocoddyl::ActionModelAbstract> action;
  switch (type) {
    case ActionModelTypes::ActionModelUnicycle:
      action = std::make_shared<crocoddyl::ActionModelUnicycle>();
      break;
    case ActionModelTypes::ActionModelLQRDriftFree:
      switch (instance) {
        case First:
          action = std::make_shared<crocoddyl::ActionModelLQR>(8, 2, true);
          break;
        case Second:
        case Terminal:
          action = std::make_shared<crocoddyl::ActionModelLQR>(8, 4, true);
          break;
      }
    case ActionModelTypes::ActionModelLQR:
      switch (instance) {
        case First:
          action = std::make_shared<crocoddyl::ActionModelLQR>(8, 2, false);
          break;
        case Second:
        case Terminal:
          action = std::make_shared<crocoddyl::ActionModelLQR>(8, 4, false);
          break;
      }
      break;
    case ActionModelTypes::ActionModelRandomLQR:
      switch (instance) {
        case First:
          action = std::make_shared<crocoddyl::ActionModelLQR>(
              crocoddyl::ActionModelLQR::Random(8, 2));
          break;
        case Second:
        case Terminal:
          action = std::make_shared<crocoddyl::ActionModelLQR>(
              crocoddyl::ActionModelLQR::Random(8, 4));
          break;
      }
      break;
    case ActionModelTypes::ActionModelRandomLQRwithTerminalConstraint:
      switch (instance) {
        case First:
          action = std::make_shared<crocoddyl::ActionModelLQR>(
              crocoddyl::ActionModelLQR::Random(8, 2));
          break;
        case Second:
          action = std::make_shared<crocoddyl::ActionModelLQR>(
              crocoddyl::ActionModelLQR::Random(8, 4));
          break;
        case Terminal:
          action = std::make_shared<crocoddyl::ActionModelLQR>(
              crocoddyl::ActionModelLQR::Random(8, 4, 0, 2));
          break;
      }
      break;
    case ActionModelTypes::ActionModelImpulseFwdDynamics_HyQ:
      action = create_impulseFwdDynamics(StateModelTypes::StateMultibody_HyQ);
      break;
    case ActionModelTypes::ActionModelImpulseFwdDynamics_Talos:
      action = create_impulseFwdDynamics(StateModelTypes::StateMultibody_Talos);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ActionModelTypes::Type given");
      break;
  }
  return action;
}

std::shared_ptr<crocoddyl::ActionModelImpulseFwdDynamics>
ActionModelFactory::create_impulseFwdDynamics(
    StateModelTypes::Type state_type) const {
  std::shared_ptr<crocoddyl::ActionModelImpulseFwdDynamics> action;
  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ImpulseModelMultiple> impulse;
  std::shared_ptr<crocoddyl::CostModelSum> cost;
  state = std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(state_type));
  impulse = std::make_shared<crocoddyl::ImpulseModelMultiple>(state);
  cost = std::make_shared<crocoddyl::CostModelSum>(state, 0);
  double r_coeff = 0.;  // TODO(cmastall): random_real_in_range(1e-16, 1e-2);
  double damping = 0.;  // TODO(cmastall): random_real_in_range(1e-16, 1e-2);

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

  switch (state_type) {
    case StateModelTypes::StateMultibody_HyQ:
      impulse->addImpulse("lf", ImpulseModelFactory().create(
                                    ImpulseModelTypes::ImpulseModel3D_LOCAL,
                                    PinocchioModelTypes::HyQ, "lf_foot"));
      impulse->addImpulse("rf", ImpulseModelFactory().create(
                                    ImpulseModelTypes::ImpulseModel3D_WORLD,
                                    PinocchioModelTypes::HyQ, "rf_foot"));
      impulse->addImpulse("lh", ImpulseModelFactory().create(
                                    ImpulseModelTypes::ImpulseModel3D_LWA,
                                    PinocchioModelTypes::HyQ, "lh_foot"));
      impulse->addImpulse("rh", ImpulseModelFactory().create(
                                    ImpulseModelTypes::ImpulseModel3D_LOCAL,
                                    PinocchioModelTypes::HyQ, "rh_foot"));

      // friction cone
      cost->addCost(
          "lf_cone",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, friction_activation,
              std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                  state, state->get_pinocchio()->getFrameId("lf_foot"),
                  friction_cone, 0)),
          0.1);
      cost->addCost(
          "rf_cone",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, friction_activation,
              std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                  state, state->get_pinocchio()->getFrameId("rf_foot"),
                  friction_cone, 0)),
          0.1);
      cost->addCost(
          "lh_cone",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, friction_activation,
              std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                  state, state->get_pinocchio()->getFrameId("lh_foot"),
                  friction_cone, 0)),
          0.1);
      cost->addCost(
          "rh_cone",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, friction_activation,
              std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                  state, state->get_pinocchio()->getFrameId("rh_foot"),
                  friction_cone, 0)),
          0.1);
      // force regularization
      cost->addCost(
          "lf_forceReg",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                         state, state->get_pinocchio()->getFrameId("lf_foot"),
                         force, 3, 0)),
          0.1);
      cost->addCost(
          "rf_forceReg",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                         state, state->get_pinocchio()->getFrameId("rf_foot"),
                         force, 3, 0)),
          0.1);
      cost->addCost(
          "lh_forceReg",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                         state, state->get_pinocchio()->getFrameId("lh_foot"),
                         force, 3, 0)),
          0.1);
      cost->addCost(
          "rh_forceReg",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, std::make_shared<crocoddyl::ResidualModelContactForce>(
                         state, state->get_pinocchio()->getFrameId("rh_foot"),
                         force, 3, 0)),
          0.1);
      break;
    case StateModelTypes::StateMultibody_Talos:
      impulse->addImpulse("lf",
                          ImpulseModelFactory().create(
                              ImpulseModelTypes::ImpulseModel6D_LOCAL,
                              PinocchioModelTypes::Talos, "left_sole_link"));
      impulse->addImpulse("rf",
                          ImpulseModelFactory().create(
                              ImpulseModelTypes::ImpulseModel6D_WORLD,
                              PinocchioModelTypes::Talos, "right_sole_link"));

      // friction / wrench cone
      cost->addCost(
          "lf_cone",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, friction_activation,
              std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                  state, state->get_pinocchio()->getFrameId("left_sole_link"),
                  friction_cone, 0)),
          0.01);
      cost->addCost(
          "rf_cone",
          std::make_shared<crocoddyl::CostModelResidual>(
              state, wrench_activation,
              std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                  state, state->get_pinocchio()->getFrameId("right_sole_link"),
                  wrench_cone, 0)),
          0.01);
      // force regularization
      cost->addCost(
          "lf_forceReg",
          std::make_shared<crocoddyl::CostModelResidual>(
              state,
              std::make_shared<crocoddyl::ResidualModelContactForce>(
                  state, state->get_pinocchio()->getFrameId("left_sole_link"),
                  force, 6, 0)),
          0.01);
      cost->addCost(
          "rf_forceReg",
          std::make_shared<crocoddyl::CostModelResidual>(
              state,
              std::make_shared<crocoddyl::ResidualModelContactForce>(
                  state, state->get_pinocchio()->getFrameId("right_sole_link"),
                  force, 6, 0)),
          0.01);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  cost->addCost("state",
                CostModelFactory().create(
                    CostModelTypes::CostModelResidualState, state_type,
                    ActivationModelTypes::ActivationModelQuad, 0),
                0.1);
  action = std::make_shared<crocoddyl::ActionModelImpulseFwdDynamics>(
      state, impulse, cost, r_coeff, damping, true);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
