///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "impulse_constraint.hpp"
#include "impulse.hpp"
#include "cost.hpp"
#include "crocoddyl/multibody/actions/impulse-fwddyn.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/multibody/residuals/impulse-com.hpp"
#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-control-gravity.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ImpulseConstraintModelTypes::Type> ImpulseConstraintModelTypes::all(
    ImpulseConstraintModelTypes::init_all());

std::ostream &operator<<(std::ostream &os, ImpulseConstraintModelTypes::Type type) {
  switch (type) {
    case ImpulseConstraintModelTypes::CostModelResidualImpulseCoMEquality:
      os << "CostModelResidualImpulseCoMEquality";
      break;
    case ImpulseConstraintModelTypes::ConstraintModelResidualImpulseForceEquality:
      os << "ConstraintModelResidualImpulseForceEquality";
      break;
    case ImpulseConstraintModelTypes::ConstraintModelResidualImpulseCoPPositionInequality:
      os << "ConstraintModelResidualImpulseCoPPositionInequality";
      break;
    case ImpulseConstraintModelTypes::ConstraintModelResidualImpulseFrictionConeInequality:
      os << "ConstraintModelResidualImpulseFrictionConeInequality";
      break;
    case ImpulseConstraintModelTypes::ConstraintModelResidualImpulseWrenchConeInequality:
      os << "ConstraintModelResidualImpulseWrenchConeInequality";
      break;
    case ImpulseConstraintModelTypes::NbImpulseConstraintModelTypes:
      os << "NbImpulseConstraintModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ImpulseConstraintModelFactory::ImpulseConstraintModelFactory() {}
ImpulseConstraintModelFactory::~ImpulseConstraintModelFactory() {}

boost::shared_ptr<crocoddyl::ActionModelAbstract> ImpulseConstraintModelFactory::create(
    ImpulseConstraintModelTypes::Type constraint_type, PinocchioModelTypes::Type model_type) const {
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
    default:
      throw_pretty(__FILE__ ": Wrong PinocchioModelTypes::Type given");
      break;
  }

  // Create impulse impulse diff-action model with standard cost functions
  boost::shared_ptr<crocoddyl::ActionModelImpulseFwdDynamics> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  boost::shared_ptr<crocoddyl::ImpulseModelMultiple> impulse;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;
  boost::shared_ptr<crocoddyl::ConstraintModelManager> constraint;
  state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
  impulse = boost::make_shared<crocoddyl::ImpulseModelMultiple>(state);
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, 0);
  constraint = boost::make_shared<crocoddyl::ConstraintModelManager>(state, 0);
  std::vector<std::size_t> frame_ids = model_factory.get_frame_ids();
  // Define the impulse model
  switch (state_type) {
    case StateModelTypes::StateMultibody_Talos:
      impulse->addImpulse(
          "lf", ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel6D, model_type, "left_sole_link"));
      impulse->addImpulse(
          "rf", ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel6D, model_type, "right_sole_link"));
      break;
    case StateModelTypes::StateMultibody_RandomHumanoid:
      impulse->addImpulse("lf",
                          ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel6D, model_type, "lleg6_body"));
      impulse->addImpulse("rf",
                          ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel6D, model_type, "rleg6_body"));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  // Define standard cost functions
  cost->addCost("state",
                CostModelFactory().create(CostModelTypes::CostModelResidualState, state_type,
                                          ActivationModelTypes::ActivationModelQuad, 0),
                0.1);

  // Define the constraint function
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  crocoddyl::FrictionCone friction_cone(R, 1.);
  crocoddyl::WrenchCone wrench_cone(R, 1., Eigen::Vector2d(0.1, 0.1));
  crocoddyl::CoPSupport cop_support(R, Eigen::Vector2d(0.1, 0.1));
  Eigen::VectorXd lb, ub;
  switch (constraint_type) {
    case ImpulseConstraintModelTypes::CostModelResidualImpulseCoMEquality:
      constraint->addConstraint("constraint",
                                boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                    state, boost::make_shared<crocoddyl::ResidualModelImpulseCoM>(state)));
      break;
    case ImpulseConstraintModelTypes::ConstraintModelResidualImpulseForceEquality:
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        constraint->addConstraint("constraint", boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                                    state, boost::make_shared<crocoddyl::ResidualModelContactForce>(
                                                               state, frame_ids[i], pinocchio::Force::Random(),
                                                               model_factory.get_contact_nc(), 0)));
      }
      break;
    case ImpulseConstraintModelTypes::ConstraintModelResidualImpulseCoPPositionInequality:
      lb = cop_support.get_lb();
      ub = cop_support.get_ub();
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        constraint->addConstraint("constraint", boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                                    state,
                                                    boost::make_shared<crocoddyl::ResidualModelContactCoPPosition>(
                                                        state, frame_ids[i], cop_support, 0),
                                                    lb, ub));
      }
      break;
    case ImpulseConstraintModelTypes::ConstraintModelResidualImpulseFrictionConeInequality:
      lb = friction_cone.get_lb();
      ub = friction_cone.get_ub();
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        constraint->addConstraint("constraint", boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                                    state,
                                                    boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                                                        state, frame_ids[i], friction_cone, 0),
                                                    lb, ub));
      }
      break;
    case ImpulseConstraintModelTypes::ConstraintModelResidualImpulseWrenchConeInequality:
      lb = wrench_cone.get_lb();
      ub = wrench_cone.get_ub();
      for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        constraint->addConstraint("constraint", boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                                    state,
                                                    boost::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                                                        state, frame_ids[i], wrench_cone, 0),
                                                    lb, ub));
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ImpulseConstraintModelTypes::Type given");
      break;
  }

  double r_coeff = 0.;  // TODO(cmastall): random_real_in_range(1e-16, 1e-2);
  double damping = 0.;  // TODO(cmastall): random_real_in_range(1e-16, 1e-2);
  action = boost::make_shared<crocoddyl::ActionModelImpulseFwdDynamics>(state, impulse, cost, constraint, r_coeff,
                                                                        damping, true);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
