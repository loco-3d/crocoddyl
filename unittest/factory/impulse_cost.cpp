///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "impulse_cost.hpp"

#include "action.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/impulse-com.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ImpulseCostModelTypes::Type> ImpulseCostModelTypes::all(
    ImpulseCostModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ImpulseCostModelTypes::Type type) {
  switch (type) {
    case ImpulseCostModelTypes::CostModelResidualImpulseCoM:
      os << "CostModelResidualImpulseCoM";
      break;
    case ImpulseCostModelTypes::CostModelResidualContactForce:
      os << "CostModelResidualContactForce";
      break;
    case ImpulseCostModelTypes::CostModelResidualContactCoPPosition:
      os << "CostModelResidualContactCoPPosition";
      break;
    case ImpulseCostModelTypes::CostModelResidualContactFrictionCone:
      os << "CostModelResidualContactFrictionCone";
      break;
    case ImpulseCostModelTypes::CostModelResidualContactWrenchCone:
      os << "CostModelResidualContactWrenchCone";
      break;
    case ImpulseCostModelTypes::NbImpulseCostModelTypes:
      os << "NbImpulseCostModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ImpulseCostModelFactory::ImpulseCostModelFactory() {}
ImpulseCostModelFactory::~ImpulseCostModelFactory() {}

std::shared_ptr<crocoddyl::ActionModelAbstract> ImpulseCostModelFactory::create(
    ImpulseCostModelTypes::Type cost_type, PinocchioModelTypes::Type model_type,
    ActivationModelTypes::Type activation_type) const {
  // Create impulse action model with no cost
  std::shared_ptr<crocoddyl::ActionModelImpulseFwdDynamics> action;
  switch (model_type) {
    case PinocchioModelTypes::Talos:
      action = ActionModelFactory().create_impulseFwdDynamics(
          StateModelTypes::StateMultibody_Talos);
      break;
    case PinocchioModelTypes::HyQ:
      action = ActionModelFactory().create_impulseFwdDynamics(
          StateModelTypes::StateMultibody_HyQ);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong PinocchioModelTypes::Type given");
      break;
  }
  action->get_costs()->removeCost("state");

  // Create cost
  std::shared_ptr<crocoddyl::CostModelAbstract> cost;
  PinocchioModelFactory model_factory(model_type);
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(action->get_state());
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  std::vector<std::size_t> frame_ids = model_factory.get_frame_ids();
  switch (cost_type) {
    case ImpulseCostModelTypes::CostModelResidualImpulseCoM:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, ActivationModelFactory().create(activation_type, 3),
          std::make_shared<crocoddyl::ResidualModelImpulseCoM>(state));
      action->get_costs()->addCost("cost_0", cost, 0.01);
      break;
    case ImpulseCostModelTypes::CostModelResidualContactForce:
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        cost = std::make_shared<crocoddyl::CostModelResidual>(
            state,
            ActivationModelFactory().create(activation_type,
                                            model_factory.get_contact_nc()),
            std::make_shared<crocoddyl::ResidualModelContactForce>(
                state, frame_ids[i], pinocchio::Force::Random(),
                model_factory.get_contact_nc(), 0));
        action->get_costs()->addCost("cost_" + std::to_string(i), cost, 0.01);
      }
      break;
    case ImpulseCostModelTypes::CostModelResidualContactCoPPosition:
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        cost = std::make_shared<crocoddyl::CostModelResidual>(
            state, ActivationModelFactory().create(activation_type, 4),
            std::make_shared<crocoddyl::ResidualModelContactCoPPosition>(
                state, frame_ids[i],
                CoPSupport(Eigen::Matrix3d::Identity(),
                           Eigen::Vector2d(0.1, 0.1)),
                0));
        action->get_costs()->addCost("cost_" + std::to_string(i), cost, 0.01);
      }
      break;
    case ImpulseCostModelTypes::CostModelResidualContactFrictionCone:
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        cost = std::make_shared<crocoddyl::CostModelResidual>(
            state, ActivationModelFactory().create(activation_type, 5),
            std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                state, frame_ids[i], crocoddyl::FrictionCone(R, 1.), 0));
        action->get_costs()->addCost("cost_" + std::to_string(i), cost, 0.01);
      }
      break;
    case ImpulseCostModelTypes::CostModelResidualContactWrenchCone:
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        cost = std::make_shared<crocoddyl::CostModelResidual>(
            state, ActivationModelFactory().create(activation_type, 17),
            std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                state, frame_ids[i],
                crocoddyl::WrenchCone(R, 1., Eigen::Vector2d(0.1, 0.1)), 0));
        action->get_costs()->addCost("cost_" + std::to_string(i), cost, 0.01);
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ImpulseCostModelTypes::Type given");
      break;
  }

  // Include the cost in the impulse model
  action->get_costs()->addCost("cost", cost, 0.1);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
