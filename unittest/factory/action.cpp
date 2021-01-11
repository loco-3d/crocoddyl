///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "action.hpp"
#include "state.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/multibody/actions/impulse-fwddyn.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
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
    case ActionModelTypes::ImpulseFwdDynamicsHyQ:
      os << "ImpulseFwdDynamicsHyQ";
      break;
    case ActionModelTypes::ImpulseFwdDynamicsTalos:
      os << "ImpulseFwdDynamicsTalos";
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
  StateModelFactory state_factory;
  boost::shared_ptr<crocoddyl::ActionModelAbstract> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation;
  boost::shared_ptr<crocoddyl::ImpulseModelMultiple> impulses;
  boost::shared_ptr<crocoddyl::CostModelSum> costs;
  double r_coeff = 0.;  // TODO(cmastall): random_real_in_range(1e-16, 1e-2);
  double damping = 0.;  // TODO(cmastall): random_real_in_range(1e-16, 1e-2);
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
    case ActionModelTypes::ImpulseFwdDynamicsHyQ:
      state = boost::static_pointer_cast<crocoddyl::StateMultibody>(
          state_factory.create(StateModelTypes::StateMultibody_HyQ));
      costs = boost::make_shared<crocoddyl::CostModelSum>(state, 0);
      costs->addCost("state", boost::make_shared<crocoddyl::CostModelState>(state, 0), 0.1);
      actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);
      impulses = boost::make_shared<crocoddyl::ImpulseModelMultiple>(state);
      impulses->addImpulse("rf_impulse", boost::make_shared<crocoddyl::ImpulseModel3D>(
                                             state, state->get_pinocchio()->getFrameId("rf_foot")));
      impulses->addImpulse("lf_impulse", boost::make_shared<crocoddyl::ImpulseModel3D>(
                                             state, state->get_pinocchio()->getFrameId("lf_foot")));
      impulses->addImpulse("rh_impulse", boost::make_shared<crocoddyl::ImpulseModel3D>(
                                             state, state->get_pinocchio()->getFrameId("rh_foot")));
      impulses->addImpulse("lh_impulse", boost::make_shared<crocoddyl::ImpulseModel3D>(
                                             state, state->get_pinocchio()->getFrameId("lh_foot")));
      action =
          boost::make_shared<crocoddyl::ActionModelImpulseFwdDynamics>(state, impulses, costs, r_coeff, damping, true);
      break;
    case ActionModelTypes::ImpulseFwdDynamicsTalos:
      state = boost::static_pointer_cast<crocoddyl::StateMultibody>(
          state_factory.create(StateModelTypes::StateMultibody_Talos));
      costs = boost::make_shared<crocoddyl::CostModelSum>(state, 0);
      costs->addCost("state", boost::make_shared<crocoddyl::CostModelState>(state, 0), 0.1);
      actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);
      impulses = boost::make_shared<crocoddyl::ImpulseModelMultiple>(state);
      impulses->addImpulse("r_sole_impulse", boost::make_shared<crocoddyl::ImpulseModel6D>(
                                                 state, state->get_pinocchio()->getFrameId("right_sole_link")));
      impulses->addImpulse("l_sole_impulse", boost::make_shared<crocoddyl::ImpulseModel6D>(
                                                 state, state->get_pinocchio()->getFrameId("left_sole_link")));
      action =
          boost::make_shared<crocoddyl::ActionModelImpulseFwdDynamics>(state, impulses, costs, r_coeff, damping, true);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ActionModelTypes::Type given");
      break;
  }
  return action;
}

}  // namespace unittest
}  // namespace crocoddyl
