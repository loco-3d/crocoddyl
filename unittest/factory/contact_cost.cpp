///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "contact_cost.hpp"
#include "diff_action.hpp"
#include "crocoddyl/multibody/costs/contact-force.hpp"
#include "crocoddyl/multibody/costs/contact-cop-position.hpp"
#include "crocoddyl/multibody/costs/contact-friction-cone.hpp"
#include "crocoddyl/multibody/costs/contact-wrench-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ContactCostModelTypes::Type> ContactCostModelTypes::all(ContactCostModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ContactCostModelTypes::Type type) {
  switch (type) {
    case ContactCostModelTypes::CostModelContactForce:
      os << "CostModelContactForce";
      break;
    case ContactCostModelTypes::CostModelContactCoPPosition:
      os << "CostModelContactCoPPosition";
      break;
    case ContactCostModelTypes::CostModelContactFrictionCone:
      os << "CostModelContactFrictionCone";
      break;
    case ContactCostModelTypes::CostModelContactWrenchCone:
      os << "CostModelContactWrenchCone";
      break;
    case ContactCostModelTypes::NbContactCostModelTypes:
      os << "NbContactCostModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ContactCostModelFactory::ContactCostModelFactory() {}
ContactCostModelFactory::~ContactCostModelFactory() {}

boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> ContactCostModelFactory::create(
    ContactCostModelTypes::Type cost_type, PinocchioModelTypes::Type model_type,
    ActivationModelTypes::Type activation_type, ActuationModelTypes::Type actuation_type) const {
  // Create contact action model with no cost
  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> action;
  switch (model_type) {
    case PinocchioModelTypes::Talos:
      action = DifferentialActionModelFactory().create_contactFwdDynamics(StateModelTypes::StateMultibody_Talos,
                                                                          actuation_type, false);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong PinocchioModelTypes::Type given");
      break;
  }
  action->get_costs()->removeCost("state");
  action->get_costs()->removeCost("control");

  // Create cost
  boost::shared_ptr<crocoddyl::CostModelAbstract> cost;
  PinocchioModelFactory model_factory(model_type);
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(action->get_state());
  const std::size_t nu = action->get_actuation()->get_nu();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  switch (cost_type) {
    case ContactCostModelTypes::CostModelContactForce:
      cost = boost::make_shared<crocoddyl::CostModelContactForce>(
          state, ActivationModelFactory().create(activation_type, 6),
          crocoddyl::FrameForce(model_factory.get_frame_id(), pinocchio::Force::Random()), nu);
      break;
    case ContactCostModelTypes::CostModelContactCoPPosition:
      cost = boost::make_shared<crocoddyl::CostModelContactCoPPosition>(
          state, ActivationModelFactory().create(activation_type, 4),
          crocoddyl::FrameCoPSupport(model_factory.get_frame_id(), Eigen::Vector2d(0.1, 0.1)), nu);
      break;
    case ContactCostModelTypes::CostModelContactFrictionCone:
      cost = boost::make_shared<crocoddyl::CostModelContactFrictionCone>(
          state, ActivationModelFactory().create(activation_type, 5),
          crocoddyl::FrameFrictionCone(model_factory.get_frame_id(),
                                       crocoddyl::FrictionCone(Eigen::Vector3d(0., 0., 1.), 1.)),
          nu);
      break;
    case ContactCostModelTypes::CostModelContactWrenchCone:
      cost = boost::make_shared<crocoddyl::CostModelContactWrenchCone>(
          state, ActivationModelFactory().create(activation_type, 17),
          crocoddyl::FrameWrenchCone(model_factory.get_frame_id(),
                                     crocoddyl::WrenchCone(R, 1., Eigen::Vector2d(0.1, 0.1))),
          nu);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ContactCostModelTypes::Type given");
      break;
  }

  // Include the cost in the contact model
  action->get_costs()->addCost("cost", cost, 0.001);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
