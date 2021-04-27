///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "impulse_cost.hpp"
#include "action.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/residuals/impulse-com.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ImpulseCostModelTypes::Type> ImpulseCostModelTypes::all(ImpulseCostModelTypes::init_all());

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
    case ImpulseCostModelTypes::CostModelResidualImpulseCoM:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state,
          ActivationModelFactory().create(activation_type, 3), boost::make_shared<crocoddyl::ResidualModelImpulseCoM>(state));
      break;
    case ImpulseCostModelTypes::CostModelResidualContactForce:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
							      state,ActivationModelFactory().create(activation_type, 6),
							      boost::make_shared<crocoddyl::ResidualModelContactForce>(
														       state,
														       model_factory.get_frame_id(), pinocchio::Force::Random(), model_factory.get_contact_nc(), 0));
      break;
    case ImpulseCostModelTypes::CostModelResidualContactCoPPosition:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state,
          ActivationModelFactory().create(activation_type, 4),
          boost::make_shared<crocoddyl::ResidualModelContactCoPPosition>(
              state, model_factory.get_frame_id(), CoPSupport(Eigen::Matrix3d::Identity(), Eigen::Vector2d(0.1, 0.1)),
              0));
      break;
    case ImpulseCostModelTypes::CostModelResidualContactFrictionCone:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state,
          ActivationModelFactory().create(activation_type, 5),
          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(state, model_factory.get_frame_id(),
                                                                          crocoddyl::FrictionCone(R, 1.), 0));
      break;
    case ImpulseCostModelTypes::CostModelResidualContactWrenchCone:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state,
          ActivationModelFactory().create(activation_type, 17),
          boost::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
              state, model_factory.get_frame_id(), crocoddyl::WrenchCone(R, 1., Eigen::Vector2d(0.1, 0.1)), 0));
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
