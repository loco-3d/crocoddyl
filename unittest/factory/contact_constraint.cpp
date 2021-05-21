///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "contact_constraint.hpp"
#include "contact.hpp"
#include "cost.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-control-gravity.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ContactConstraintModelTypes::Type> ContactConstraintModelTypes::all(
    ContactConstraintModelTypes::init_all());

std::ostream &operator<<(std::ostream &os, ContactConstraintModelTypes::Type type) {
  switch (type) {
    case ContactConstraintModelTypes::ConstraintModelResidualContactForceEquality:
      os << "ConstraintModelResidualContactForceEquality";
      break;
    case ContactConstraintModelTypes::ConstraintModelResidualContactCoPPositionInequality:
      os << "ConstraintModelResidualContactCoPPositionInequality";
      break;
    case ContactConstraintModelTypes::ConstraintModelResidualContactFrictionConeInequality:
      os << "ConstraintModelResidualContactFrictionConeInequality";
      break;
    case ContactConstraintModelTypes::ConstraintModelResidualContactWrenchConeInequality:
      os << "ConstraintModelResidualContactWrenchConeInequality";
      break;
    case ContactConstraintModelTypes::ConstraintModelResidualContactControlGravInequality:
      os << "ConstraintModelResidualContactControlGravInequality";
      break;
    case ContactConstraintModelTypes::NbContactConstraintModelTypes:
      os << "NbContactConstraintModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ContactConstraintModelFactory::ContactConstraintModelFactory() {}
ContactConstraintModelFactory::~ContactConstraintModelFactory() {}

boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> ContactConstraintModelFactory::create(
    ContactConstraintModelTypes::Type constraint_type, PinocchioModelTypes::Type model_type,
    ActuationModelTypes::Type actuation_type) const {
  // Identify the state type given the model type
  StateModelTypes::Type state_type;
  switch (model_type) {
    case PinocchioModelTypes::Talos:
      state_type = StateModelTypes::StateMultibody_Talos;
      break;
    case PinocchioModelTypes::RandomHumanoid:
      state_type = StateModelTypes::StateMultibody_RandomHumanoid;
      break;
    default:
      throw_pretty(__FILE__ ": Wrong PinocchioModelTypes::Type given");
      break;
  }

  // Create contact contact diff-action model with standard cost functions
  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contact;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;
  boost::shared_ptr<crocoddyl::ConstraintModelManager> constraint;
  state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  contact = boost::make_shared<crocoddyl::ContactModelMultiple>(state, actuation->get_nu());
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  constraint = boost::make_shared<crocoddyl::ConstraintModelManager>(state, actuation->get_nu());
  // Define the contact model
  switch (state_type) {
    case StateModelTypes::StateMultibody_Talos:
      contact->addContact("lf", ContactModelFactory().create(ContactModelTypes::ContactModel6D, model_type,
                                                             "left_sole_link", actuation->get_nu()));
      contact->addContact("rf", ContactModelFactory().create(ContactModelTypes::ContactModel6D, model_type,
                                                             "right_sole_link", actuation->get_nu()));
      break;
    case StateModelTypes::StateMultibody_RandomHumanoid:
      contact->addContact("lf", ContactModelFactory().create(ContactModelTypes::ContactModel6D, model_type,
                                                             "lleg6_body", actuation->get_nu()));
      contact->addContact("rf", ContactModelFactory().create(ContactModelTypes::ContactModel6D, model_type,
                                                             "rleg6_body", actuation->get_nu()));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  // Define standard cost functions
  cost->addCost("state",
                CostModelFactory().create(CostModelTypes::CostModelResidualState, state_type,
                                          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
                0.1);
  cost->addCost("control",
                CostModelFactory().create(CostModelTypes::CostModelResidualControl, state_type,
                                          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
                0.1);

  // Define the constraint function
  PinocchioModelFactory model_factory(model_type);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  crocoddyl::FrictionCone friction_cone(R, 1.);
  crocoddyl::WrenchCone wrench_cone(R, 1., Eigen::Vector2d(0.1, 0.1));
  crocoddyl::CoPSupport cop_support(R, Eigen::Vector2d(0.1, 0.1));
  Eigen::VectorXd lb, ub;
  switch (constraint_type) {
    case ContactConstraintModelTypes::ConstraintModelResidualContactForceEquality:
      constraint->addConstraint("constraint",
                                boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                    state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                                               state, model_factory.get_frame_id(), pinocchio::Force::Random(),
                                               model_factory.get_contact_nc(), actuation->get_nu())));
      break;
    case ContactConstraintModelTypes::ConstraintModelResidualContactCoPPositionInequality:
      lb = cop_support.get_lb();
      ub = cop_support.get_ub();
      constraint->addConstraint("constraint",
                                boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                    state,
                                    boost::make_shared<crocoddyl::ResidualModelContactCoPPosition>(
                                        state, model_factory.get_frame_id(), cop_support, actuation->get_nu()),
                                    lb, ub));
      break;
    case ContactConstraintModelTypes::ConstraintModelResidualContactFrictionConeInequality:
      lb = friction_cone.get_lb();
      ub = friction_cone.get_ub();
      constraint->addConstraint("constraint",
                                boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                    state,
                                    boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                                        state, model_factory.get_frame_id(), friction_cone, actuation->get_nu()),
                                    lb, ub),
                                true);
      break;
    case ContactConstraintModelTypes::ConstraintModelResidualContactWrenchConeInequality:
      lb = wrench_cone.get_lb();
      ub = wrench_cone.get_ub();
      constraint->addConstraint("constraint",
                                boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                    state,
                                    boost::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                                        state, model_factory.get_frame_id(), wrench_cone, actuation->get_nu()),
                                    lb, ub),
                                true);
      break;
    case ContactConstraintModelTypes::ConstraintModelResidualContactControlGravInequality:
      lb = Eigen::VectorXd::Zero(state->get_nv());
      ub = Eigen::VectorXd::Random(state->get_nv()).cwiseAbs();
      constraint->addConstraint(
          "constraint",
          boost::make_shared<crocoddyl::ConstraintModelResidual>(
              state, boost::make_shared<crocoddyl::ResidualModelContactControlGrav>(state, actuation->get_nu()), lb,
              ub));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ContactConstraintModelTypes::Type given");
      break;
  }

  action = boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state, actuation, contact, cost,
                                                                                    constraint, 0., true);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
