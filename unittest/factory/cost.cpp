///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "cost.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/core/costs/control.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"
#include "crocoddyl/multibody/costs/control-gravity.hpp"
#include "crocoddyl/multibody/costs/control-gravity-contact.hpp"
// #include "crocoddyl/multibody/costs/centroidal-momentum.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"
#include "crocoddyl/multibody/costs/frame-velocity.hpp"
#include "crocoddyl/multibody/costs/contact-friction-cone.hpp"
#include "crocoddyl/multibody/costs/contact-wrench-cone.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<CostModelTypes::Type> CostModelTypes::all(CostModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, CostModelTypes::Type type) {
  switch (type) {
    case CostModelTypes::CostModelState:
      os << "CostModelState";
      break;
    case CostModelTypes::CostModelControl:
      os << "CostModelControl";
      break;
    case CostModelTypes::CostModelCoMPosition:
      os << "CostModelCoMPosition";
      break;
    case CostModelTypes::CostModelControlGrav:
      os << "CostModelControlGrav";
      break;
    case CostModelTypes::CostModelControlGravContact:
      os << "CostModelControlGravContact";
      break;
    // case CostModelTypes::CostModelCentroidalMomentum:
    //   os << "CostModelCentroidalMomentum";
    //   break;
    case CostModelTypes::CostModelFramePlacement:
      os << "CostModelFramePlacement";
      break;
    case CostModelTypes::CostModelFrameRotation:
      os << "CostModelFrameRotation";
      break;
    case CostModelTypes::CostModelFrameTranslation:
      os << "CostModelFrameTranslation";
      break;
    case CostModelTypes::CostModelFrameVelocity:
      os << "CostModelFrameVelocity";
      break;
    case CostModelTypes::NbCostModelTypes:
      os << "NbCostModelTypes";
      break;
    default:
      break;
  }
  return os;
}

CostModelFactory::CostModelFactory() {}
CostModelFactory::~CostModelFactory() {}

boost::shared_ptr<crocoddyl::CostModelAbstract> CostModelFactory::create(CostModelTypes::Type cost_type,
                                                                         StateModelTypes::Type state_type,
                                                                         ActivationModelTypes::Type activation_type,
                                                                         std::size_t nu) const {
  StateModelFactory state_factory;
  ActivationModelFactory activation_factory;
  boost::shared_ptr<crocoddyl::CostModelAbstract> cost;
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(state_type));
  //boost::shared_ptr<crocoddyl::ActuationModelFull> actuation_full =
  //    boost::make_shared<crocoddyl::ActuationModelFull>(state);
  boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation_floatbase =
      boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);   
  crocoddyl::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (cost_type) {
    case CostModelTypes::CostModelState:
      cost = boost::make_shared<crocoddyl::CostModelState>(
          state, activation_factory.create(activation_type, state->get_ndx()), state->rand(), nu);
      break;
    case CostModelTypes::CostModelControl:
      cost = boost::make_shared<crocoddyl::CostModelControl>(state, activation_factory.create(activation_type, nu),
                                                             Eigen::VectorXd::Random(nu));
      break;
    case CostModelTypes::CostModelCoMPosition:
      cost = boost::make_shared<crocoddyl::CostModelCoMPosition>(state, activation_factory.create(activation_type, 3),
                                                                 Eigen::Vector3d::Random(), nu);
      break;
    //case CostModelTypes::CostModelControlGrav:
    //  cost = boost::make_shared<crocoddyl::CostModelControlGrav>(state, activation_factory.create(activation_type, nu),actuation_full);
    //  break;
    case CostModelTypes::CostModelControlGravContact:
      cost = boost::make_shared<crocoddyl::CostModelControlGravContact>(state, activation_factory.create(activation_type, nu),actuation_floatbase);
      break;
    // case CostModelTypes::CostModelCentroidalMomentum:
    //   cost = boost::make_shared<crocoddyl::CostModelCentroidalMomentum>(state_,
    //                                                                      activation_factory.create(activation_type,
    //                                                                      6), Vector6d::Random(), nu);
    //   break;
    case CostModelTypes::CostModelFramePlacement:
      cost = boost::make_shared<crocoddyl::CostModelFramePlacement>(
          state, activation_factory.create(activation_type, 6), crocoddyl::FramePlacement(frame_index, frame_SE3), nu);
      break;
    case CostModelTypes::CostModelFrameRotation:
      cost = boost::make_shared<crocoddyl::CostModelFrameRotation>(
          state, activation_factory.create(activation_type, 3),
          crocoddyl::FrameRotation(frame_index, frame_SE3.rotation()), nu);
      break;
    case CostModelTypes::CostModelFrameTranslation:
      cost = boost::make_shared<crocoddyl::CostModelFrameTranslation>(
          state, activation_factory.create(activation_type, 3),
          crocoddyl::FrameTranslation(frame_index, frame_SE3.translation()), nu);
      break;
    case CostModelTypes::CostModelFrameVelocity:
      cost = boost::make_shared<crocoddyl::CostModelFrameVelocity>(
          state, activation_factory.create(activation_type, 6),
          crocoddyl::FrameMotion(frame_index, pinocchio::Motion::Random()), nu);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong CostModelTypes::Type given");
      break;
  }
  return cost;
}

boost::shared_ptr<crocoddyl::CostModelAbstract> create_random_cost(StateModelTypes::Type state_type) {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }

  CostModelFactory factory;
  CostModelTypes::Type rand_type = static_cast<CostModelTypes::Type>(rand() % CostModelTypes::NbCostModelTypes);
  return factory.create(rand_type, state_type, ActivationModelTypes::ActivationModelQuad);
}

}  // namespace unittest
}  // namespace crocoddyl
