///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "impulse_cost.hpp"
#include "action.hpp"
#include "crocoddyl/multibody/costs/impulse-com.hpp"
#include "crocoddyl/multibody/costs/contact-impulse.hpp"
#include "crocoddyl/multibody/costs/impulse-cop-position.hpp"
#include "crocoddyl/multibody/costs/impulse-friction-cone.hpp"
#include "crocoddyl/multibody/costs/impulse-wrench-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ImpulseCostModelTypes::Type> ImpulseCostModelTypes::all(ImpulseCostModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ImpulseCostModelTypes::Type type) {
  switch (type) {
    case ImpulseCostModelTypes::CostModelImpulseCoM:
      os << "CostModelImpulseCoM";
      break;
    case ImpulseCostModelTypes::CostModelContactImpulse:
      os << "CostModelContactImpulse";
      break;
    case ImpulseCostModelTypes::CostModelImpulseCoPPosition:
      os << "CostModelImpulseCoPPosition";
      break;
    case ImpulseCostModelTypes::CostModelImpulseFrictionCone:
      os << "CostModelImpulseFrictionCone";
      break;
    case ImpulseCostModelTypes::CostModelImpulseWrenchCone:
      os << "CostModelImpulseWrenchCone";
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

boost::shared_ptr<crocoddyl::ActionModelAbstract> ImpulseCostModelFactory::create(
    ImpulseCostModelTypes::Type cost_type, PinocchioModelTypes::Type model_type,
    ActivationModelTypes::Type activation_type) const {
  // Create impulse action model with no cost
  boost::shared_ptr<crocoddyl::ActionModelImpulseFwdDynamics> action;
  switch (model_type) {
    case PinocchioModelTypes::Talos:
      action = ActionModelFactory().create_impulseFwdDynamics(StateModelTypes::StateMultibody_Talos);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong PinocchioModelTypes::Type given");
      break;
  }
  action->get_costs()->removeCost("state");

  // Create cost
  boost::shared_ptr<crocoddyl::CostModelAbstract> cost;
  PinocchioModelFactory model_factory(model_type);
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(action->get_state());
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  switch (cost_type) {
    case ImpulseCostModelTypes::CostModelImpulseCoM:
      cost = boost::make_shared<crocoddyl::CostModelImpulseCoM>(state,
                                                                ActivationModelFactory().create(activation_type, 3));
      break;
    case ImpulseCostModelTypes::CostModelContactImpulse:
      cost = boost::make_shared<crocoddyl::CostModelContactImpulse>(
          state, ActivationModelFactory().create(activation_type, 6),
          crocoddyl::FrameForce(model_factory.get_frame_id(), pinocchio::Force::Random()));
      break;
    case ImpulseCostModelTypes::CostModelImpulseCoPPosition:
      cost = boost::make_shared<crocoddyl::CostModelImpulseCoPPosition>(
          state, ActivationModelFactory().create(activation_type, 4),
          crocoddyl::FrameCoPSupport(model_factory.get_frame_id(), Eigen::Vector2d(0.1, 0.1)));
      break;
    case ImpulseCostModelTypes::CostModelImpulseFrictionCone:
      cost = boost::make_shared<crocoddyl::CostModelImpulseFrictionCone>(
          state, ActivationModelFactory().create(activation_type, 5),
          crocoddyl::FrameFrictionCone(model_factory.get_frame_id(), crocoddyl::FrictionCone(R, 1.)));
      break;
    case ImpulseCostModelTypes::CostModelImpulseWrenchCone:
      cost = boost::make_shared<crocoddyl::CostModelImpulseWrenchCone>(
          state, ActivationModelFactory().create(activation_type, 17),
          crocoddyl::FrameWrenchCone(model_factory.get_frame_id(),
                                     crocoddyl::WrenchCone(R, 1., Eigen::Vector2d(0.1, 0.1))));
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
