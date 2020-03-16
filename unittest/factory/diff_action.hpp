///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "state.hpp"
#include "actuation.hpp"
#include "cost.hpp"
#include "contact.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#ifndef CROCODDYL_ACTION_FACTORY_HPP_
#define CROCODDYL_ACTION_FACTORY_HPP_

namespace crocoddyl {
namespace unittest {

struct DifferentialActionModelTypes {
  enum Type {
    DifferentialActionModelLQR,
    DifferentialActionModelLQRDriftFree,
    DifferentialActionModelFreeFwdDynamics_TalosArm,
    DifferentialActionModelContactFwdDynamics_HyQ,
    // DifferentialActionModelContactFwdDynamics_Talos,
    DifferentialActionModelContactFwdDynamicsWithFriction_HyQ,
    // DifferentialActionModelContactFwdDynamicsWithFriction_Talos,
    NbDifferentialActionModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbDifferentialActionModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<DifferentialActionModelTypes::Type> DifferentialActionModelTypes::all(
    DifferentialActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, DifferentialActionModelTypes::Type type) {
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      os << "DifferentialActionModelLQR";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      os << "DifferentialActionModelLQRDriftFree";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm:
      os << "DifferentialActionModelFreeFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_HyQ:
      os << "DifferentialActionModelContactFwdDynamics_HyQ";
      break;
    // todo(cmastalli): investigate why the error is 10e-3
    // case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_Talos:
    //   os << "DifferentialActionModelContactFwdDynamics_Talos";
    //   break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_HyQ:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_HyQ";
      break;
    // todo(cmastalli): investigate why the error is 10e-3
    // case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_Talos:
    //   os << "DifferentialActionModelContactFwdDynamicsWithFriction_Talos";
    //   break;
    case DifferentialActionModelTypes::NbDifferentialActionModelTypes:
      os << "NbDifferentialActionModelTypes";
      break;
    default:
      break;
  }
  return os;
}

class DifferentialActionModelFactory {
 public:
  explicit DifferentialActionModelFactory() {}
  ~DifferentialActionModelFactory() {}

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create(DifferentialActionModelTypes::Type type) {
    boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> action;
    switch (type) {
      case DifferentialActionModelTypes::DifferentialActionModelLQR:
        action = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40, false);
        break;
      case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
        action = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40, true);
        break;
      case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm:
        action = create_freeFwdDynamics(StateModelTypes::StateMultibody_TalosArm);
        break;
      case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_HyQ:
        action = create_contactFwdDynamics(StateModelTypes::StateMultibody_HyQ, false);
        break;
        //   case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_Talos:
        //     action = create_contactFwdDynamics(StateModelTypes::StateMultibody_Talos, false);
        //     break;
      case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_HyQ:
        action = create_contactFwdDynamics(StateModelTypes::StateMultibody_HyQ);
        break;
        //   case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_Talos:
        //     action = create_contactFwdDynamics(StateModelTypes::StateMultibody_Talos);
        //     break;
      default:
        throw_pretty(__FILE__ ": Wrong DifferentialActionModelTypes::Type given");
        break;
    }
    return action;
  }

 private:
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create_freeFwdDynamics(
      StateModelTypes::Type state_type) {
    boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> action;
    boost::shared_ptr<crocoddyl::StateMultibody> state;
    boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
    boost::shared_ptr<crocoddyl::CostModelSum> cost;
    state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
    actuation = ActuationModelFactory().create(ActuationModelTypes::ActuationModelFull, state_type);
    cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
    cost->addCost("state",
                  CostModelFactory().create(CostModelTypes::CostModelState, state_type,
                                            ActivationModelTypes::ActivationModelQuad),
                  1.);
    cost->addCost("control",
                  CostModelFactory().create(CostModelTypes::CostModelControl, state_type,
                                            ActivationModelTypes::ActivationModelQuad),
                  1.);
    cost->addCost("frame",
                  CostModelFactory().create(CostModelTypes::CostModelFramePlacement, state_type,
                                            ActivationModelTypes::ActivationModelQuad),
                  1.);
    action = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, cost);
    return action;
  }

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create_contactFwdDynamics(
      StateModelTypes::Type state_type, bool with_friction = true) {
    boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> action;
    boost::shared_ptr<crocoddyl::StateMultibody> state;
    boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation;
    boost::shared_ptr<crocoddyl::ContactModelMultiple> contact;
    boost::shared_ptr<crocoddyl::CostModelSum> cost;
    state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
    actuation = boost::static_pointer_cast<crocoddyl::ActuationModelFloatingBase>(
        ActuationModelFactory().create(ActuationModelTypes::ActuationModelFloatingBase, state_type));
    contact = boost::make_shared<crocoddyl::ContactModelMultiple>(state, actuation->get_nu());
    cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

    crocoddyl::FrictionCone cone(Eigen::Vector3d(0, 0, 1), 0.8, 4, false);
    crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
    boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation =
        boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
    switch (state_type) {
      case StateModelTypes::StateMultibody_HyQ:
        contact->addContact("lf", boost::make_shared<crocoddyl::ContactModel3D>(
                                      state,
                                      crocoddyl::FrameTranslation(state->get_pinocchio()->getFrameId("lf_foot"),
                                                                  Eigen::Vector3d::Zero()),
                                      actuation->get_nu()));
        contact->addContact("rf", boost::make_shared<crocoddyl::ContactModel3D>(
                                      state,
                                      crocoddyl::FrameTranslation(state->get_pinocchio()->getFrameId("rf_foot"),
                                                                  Eigen::Vector3d::Zero()),
                                      actuation->get_nu()));
        contact->addContact("lh", boost::make_shared<crocoddyl::ContactModel3D>(
                                      state,
                                      crocoddyl::FrameTranslation(state->get_pinocchio()->getFrameId("lh_foot"),
                                                                  Eigen::Vector3d::Zero()),
                                      actuation->get_nu()));
        contact->addContact("rh", boost::make_shared<crocoddyl::ContactModel3D>(
                                      state,
                                      crocoddyl::FrameTranslation(state->get_pinocchio()->getFrameId("rh_foot"),
                                                                  Eigen::Vector3d::Zero()),
                                      actuation->get_nu()));
        if (with_friction) {
          cost->addCost(
              "lf_cone",
              boost::make_shared<crocoddyl::CostModelContactFrictionCone>(
                  state, activation, cone, state->get_pinocchio()->getFrameId("lf_foot"), actuation->get_nu()),
              0.1);
          cost->addCost(
              "rf_cone",
              boost::make_shared<crocoddyl::CostModelContactFrictionCone>(
                  state, activation, cone, state->get_pinocchio()->getFrameId("rf_foot"), actuation->get_nu()),
              0.1);
          cost->addCost(
              "lh_cone",
              boost::make_shared<crocoddyl::CostModelContactFrictionCone>(
                  state, activation, cone, state->get_pinocchio()->getFrameId("lh_foot"), actuation->get_nu()),
              0.1);
          cost->addCost(
              "rh_cone",
              boost::make_shared<crocoddyl::CostModelContactFrictionCone>(
                  state, activation, cone, state->get_pinocchio()->getFrameId("rh_foot"), actuation->get_nu()),
              0.1);
        }
        break;
      case StateModelTypes::StateMultibody_Talos:
        contact->addContact("lf", boost::make_shared<crocoddyl::ContactModel6D>(
                                      state,
                                      crocoddyl::FramePlacement(state->get_pinocchio()->getFrameId("left_sole_link"),
                                                                pinocchio::SE3().Identity()),
                                      actuation->get_nu()));
        contact->addContact("rf", boost::make_shared<crocoddyl::ContactModel6D>(
                                      state,
                                      crocoddyl::FramePlacement(state->get_pinocchio()->getFrameId("right_sole_link"),
                                                                pinocchio::SE3().Identity()),
                                      actuation->get_nu()));
        if (with_friction) {
          cost->addCost(
              "lf_cone",
              boost::make_shared<crocoddyl::CostModelContactFrictionCone>(
                  state, activation, cone, state->get_pinocchio()->getFrameId("left_sole_link"), actuation->get_nu()),
              0.01);
          cost->addCost(
              "rf_cone",
              boost::make_shared<crocoddyl::CostModelContactFrictionCone>(
                  state, activation, cone, state->get_pinocchio()->getFrameId("right_sole_link"), actuation->get_nu()),
              0.01);
        }
        break;
      default:
        throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
        break;
    }
    cost->addCost("state",
                  CostModelFactory().create(CostModelTypes::CostModelState, state_type,
                                            ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
                  1.);
    cost->addCost("control",
                  CostModelFactory().create(CostModelTypes::CostModelControl, state_type,
                                            ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
                  1.);
    action = boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state, actuation, contact, cost,
                                                                                      0., true);
    return action;
  }
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_ACTION_FACTORY_HPP_
