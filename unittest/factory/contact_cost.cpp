///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "contact_cost.hpp"
#include "diff_action.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-control-gravity.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ContactCostModelTypes::Type> ContactCostModelTypes::all(ContactCostModelTypes::init_all());

std::ostream &operator<<(std::ostream &os, ContactCostModelTypes::Type type) {
  switch (type) {
    case ContactCostModelTypes::CostModelResidualContactForce:
      os << "CostModelResidualContactForce";
      break;
    case ContactCostModelTypes::CostModelResidualContactCoPPosition:
      os << "CostModelResidualContactCoPPosition";
      break;
    case ContactCostModelTypes::CostModelResidualContactFrictionCone:
      os << "CostModelResidualContactFrictionCone";
      break;
    case ContactCostModelTypes::CostModelResidualContactWrenchCone:
      os << "CostModelResidualContactWrenchCone";
      break;
    case ContactCostModelTypes::CostModelResidualContactControlGrav:
      os << "CostModelResidualContactControlGrav";
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
    case ContactCostModelTypes::CostModelResidualContactForce:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state, ActivationModelFactory().create(activation_type, 6),
          boost::make_shared<crocoddyl::ResidualModelContactForce>(
              state, model_factory.get_frame_id(), pinocchio::Force::Random(), model_factory.get_contact_nc(), nu));
      break;
    case ContactCostModelTypes::CostModelResidualContactCoPPosition:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state, ActivationModelFactory().create(activation_type, 4),
          boost::make_shared<crocoddyl::ResidualModelContactCoPPosition>(
              state, model_factory.get_frame_id(), crocoddyl::CoPSupport(R, Eigen::Vector2d(0.1, 0.1)), nu));
      break;
    case ContactCostModelTypes::CostModelResidualContactFrictionCone:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state, ActivationModelFactory().create(activation_type, 5),
          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(state, model_factory.get_frame_id(),
                                                                          crocoddyl::FrictionCone(R, 1.), nu));
      break;
    case ContactCostModelTypes::CostModelResidualContactWrenchCone:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state, ActivationModelFactory().create(activation_type, 17),
          boost::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
              state, model_factory.get_frame_id(), crocoddyl::WrenchCone(R, 1., Eigen::Vector2d(0.1, 0.1)), nu));
      break;
    case ContactCostModelTypes::CostModelResidualContactControlGrav:
      cost = boost::make_shared<crocoddyl::CostModelResidual>(
          state, ActivationModelFactory().create(activation_type, state->get_nv()),
          boost::make_shared<crocoddyl::ResidualModelContactControlGrav>(state, nu));
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
