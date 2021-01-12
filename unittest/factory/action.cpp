///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2021, University of Edinburgh
// Copyright (C) 2018-2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "action.hpp"
#include "cost.hpp"
#include "impulse.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ActionModelTypes::Type> ActionModelTypes::all(ActionModelTypes::init_all());

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
    case ActionModelTypes::ImpulseFwdDynamics_HyQ:
      os << "ImpulseFwdDynamics_HyQ";
      break;
    case ActionModelTypes::ImpulseFwdDynamics_Talos:
      os << "ImpulseFwdDynamics_Talos";
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

boost::shared_ptr<crocoddyl::ActionModelAbstract> ActionModelFactory::create(ActionModelTypes::Type type,
                                                                             bool secondInstance) const {
  boost::shared_ptr<crocoddyl::ActionModelAbstract> action;
  switch (type) {
    case ActionModelTypes::ActionModelUnicycle:
      action = boost::make_shared<crocoddyl::ActionModelUnicycle>();
      break;
    case ActionModelTypes::ActionModelLQRDriftFree:
      if (secondInstance) {
        action = boost::make_shared<crocoddyl::ActionModelLQR>(80, 40, true);
      } else {
        action = boost::make_shared<crocoddyl::ActionModelLQR>(80, 20, true);
      }
      break;
    case ActionModelTypes::ActionModelLQR:
      if (secondInstance) {
        action = boost::make_shared<crocoddyl::ActionModelLQR>(80, 40, false);
      } else {
        action = boost::make_shared<crocoddyl::ActionModelLQR>(80, 20, false);
      }
      break;
    case ActionModelTypes::ImpulseFwdDynamics_HyQ:
      action = create_impulseFwdDynamics(StateModelTypes::StateMultibody_HyQ);
      break;
    case ActionModelTypes::ImpulseFwdDynamics_Talos:
      action = create_impulseFwdDynamics(StateModelTypes::StateMultibody_Talos);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ActionModelTypes::Type given");
      break;
  }
  return action;
}

boost::shared_ptr<crocoddyl::ActionModelImpulseFwdDynamics> ActionModelFactory::create_impulseFwdDynamics(
    StateModelTypes::Type state_type) const {
  boost::shared_ptr<crocoddyl::ActionModelImpulseFwdDynamics> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ImpulseModelMultiple> impulse;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;
  state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
  impulse = boost::make_shared<crocoddyl::ImpulseModelMultiple>(state);
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, 0);
  double r_coeff = 0.;  // TODO(cmastall): random_real_in_range(1e-16, 1e-2);
  double damping = 0.;  // TODO(cmastall): random_real_in_range(1e-16, 1e-2);

  switch (state_type) {
    case StateModelTypes::StateMultibody_HyQ:
      impulse->addImpulse(
          "lf", ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel3D, PinocchioModelTypes::HyQ, "lf_foot"));
      impulse->addImpulse(
          "rf", ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel3D, PinocchioModelTypes::HyQ, "rf_foot"));
      impulse->addImpulse(
          "lh", ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel3D, PinocchioModelTypes::HyQ, "lh_foot"));
      impulse->addImpulse(
          "rh", ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel3D, PinocchioModelTypes::HyQ, "rh_foot"));
      break;
    case StateModelTypes::StateMultibody_Talos:
      impulse->addImpulse("lf", ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel6D,
                                                             PinocchioModelTypes::Talos, "left_sole_link"));
      impulse->addImpulse("rf", ImpulseModelFactory().create(ImpulseModelTypes::ImpulseModel6D,
                                                             PinocchioModelTypes::Talos, "right_sole_link"));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  cost->addCost("state",
                CostModelFactory().create(CostModelTypes::CostModelState, state_type,
                                          ActivationModelTypes::ActivationModelQuad, 0),
                0.1);
  action = boost::make_shared<crocoddyl::ActionModelImpulseFwdDynamics>(state, impulse, cost, r_coeff, damping, true);
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
