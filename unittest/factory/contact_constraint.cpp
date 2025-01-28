///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "contact_constraint.hpp"

#include "contact.hpp"
#include "cost.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/residuals/contact-control-gravity.hpp"
#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ContactConstraintModelTypes::Type>
    ContactConstraintModelTypes::all(ContactConstraintModelTypes::init_all());

std::ostream &operator<<(std::ostream &os,
                         ContactConstraintModelTypes::Type type) {
  switch (type) {
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactForceEquality:
      os << "ConstraintModelResidualContactForceEquality";
      break;
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactCoPPositionInequality:
      os << "ConstraintModelResidualContactCoPPositionInequality";
      break;
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactFrictionConeInequality:
      os << "ConstraintModelResidualContactFrictionConeInequality";
      break;
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactWrenchConeInequality:
      os << "ConstraintModelResidualContactWrenchConeInequality";
      break;
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactControlGravInequality:
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

std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>
ContactConstraintModelFactory::create(
    ContactConstraintModelTypes::Type constraint_type,
    PinocchioModelTypes::Type model_type,
    ActuationModelTypes::Type actuation_type) const {
  // Identify the state type given the model type
  StateModelTypes::Type state_type;
  PinocchioModelFactory model_factory(model_type);
  switch (model_type) {
    case PinocchioModelTypes::Talos:
      state_type = StateModelTypes::StateMultibody_Talos;
      break;
    case PinocchioModelTypes::RandomHumanoid:
      state_type = StateModelTypes::StateMultibody_RandomHumanoid;
      break;
    case PinocchioModelTypes::HyQ:
      state_type = StateModelTypes::StateMultibody_HyQ;
      break;
    default:
      throw_pretty(__FILE__ ": Wrong PinocchioModelTypes::Type given");
      break;
  }

  // Create contact contact diff-action model with standard cost functions
  std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> action;
  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  std::shared_ptr<crocoddyl::ContactModelMultiple> contact;
  std::shared_ptr<crocoddyl::CostModelSum> cost;
  std::shared_ptr<crocoddyl::ConstraintModelManager> constraint;
  state = std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  contact = std::make_shared<crocoddyl::ContactModelMultiple>(
      state, actuation->get_nu());
  cost = std::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  constraint = std::make_shared<crocoddyl::ConstraintModelManager>(
      state, actuation->get_nu());
  std::vector<std::size_t> frame_ids = model_factory.get_frame_ids();
  std::vector<std::string> frame_names = model_factory.get_frame_names();
  // Define the contact model
  switch (state_type) {
    case StateModelTypes::StateMultibody_Talos:
    case StateModelTypes::StateMultibody_RandomHumanoid:
      for (std::size_t i = 0; i < frame_names.size(); ++i) {
        contact->addContact(frame_names[i],
                            ContactModelFactory().create(
                                ContactModelTypes::ContactModel6D_LOCAL,
                                model_type, Eigen::Vector2d::Random(),
                                frame_names[i], actuation->get_nu()));
      }
      break;
    case StateModelTypes::StateMultibody_HyQ:
      for (std::size_t i = 0; i < frame_names.size(); ++i) {
        contact->addContact(frame_names[i],
                            ContactModelFactory().create(
                                ContactModelTypes::ContactModel3D_LOCAL,
                                model_type, Eigen::Vector2d::Random(),
                                frame_names[i], actuation->get_nu()));
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  // Define standard cost functions
  cost->addCost(
      "state",
      CostModelFactory().create(
          CostModelTypes::CostModelResidualState, state_type,
          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
      0.1);
  cost->addCost(
      "control",
      CostModelFactory().create(
          CostModelTypes::CostModelResidualControl, state_type,
          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
      0.1);

  // Define the constraint function
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  crocoddyl::FrictionCone friction_cone(R, 1.);
  crocoddyl::WrenchCone wrench_cone(R, 1., Eigen::Vector2d(0.1, 0.1));
  crocoddyl::CoPSupport cop_support(R, Eigen::Vector2d(0.1, 0.1));
  Eigen::VectorXd lb, ub;
  switch (constraint_type) {
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactForceEquality:
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        constraint->addConstraint(
            "constraint_" + std::to_string(i),
            std::make_shared<crocoddyl::ConstraintModelResidual>(
                state,
                std::make_shared<crocoddyl::ResidualModelContactForce>(
                    state, frame_ids[i], pinocchio::Force::Random(),
                    model_factory.get_contact_nc(), actuation->get_nu())));
      }
      break;
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactCoPPositionInequality:
      lb = cop_support.get_lb();
      ub = cop_support.get_ub();
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        constraint->addConstraint(
            "constraint_" + std::to_string(i),
            std::make_shared<crocoddyl::ConstraintModelResidual>(
                state,
                std::make_shared<crocoddyl::ResidualModelContactCoPPosition>(
                    state, frame_ids[i], cop_support, actuation->get_nu()),
                lb, ub));
      }
      break;
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactFrictionConeInequality:
      lb = friction_cone.get_lb();
      ub = friction_cone.get_ub();
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        constraint->addConstraint(
            "constraint_" + std::to_string(i),
            std::make_shared<crocoddyl::ConstraintModelResidual>(
                state,
                std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, frame_ids[i], friction_cone, actuation->get_nu()),
                lb, ub),
            true);
      }
      break;
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactWrenchConeInequality:
      lb = wrench_cone.get_lb();
      ub = wrench_cone.get_ub();
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        constraint->addConstraint(
            "constraint_" + std::to_string(i),
            std::make_shared<crocoddyl::ConstraintModelResidual>(
                state,
                std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                    state, frame_ids[i], wrench_cone, actuation->get_nu()),
                lb, ub),
            true);
      }
      break;
    case ContactConstraintModelTypes::
        ConstraintModelResidualContactControlGravInequality:
      lb = Eigen::VectorXd::Zero(state->get_nv());
      ub = Eigen::VectorXd::Random(state->get_nv()).cwiseAbs();
      constraint->addConstraint(
          "constraint_0",
          std::make_shared<crocoddyl::ConstraintModelResidual>(
              state,
              std::make_shared<crocoddyl::ResidualModelContactControlGrav>(
                  state, actuation->get_nu()),
              lb, ub));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ContactConstraintModelTypes::Type given");
      break;
  }

  action =
      std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(
          state, actuation, contact, cost, constraint, 0., true);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
