///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/costs/control.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"
// #include "crocoddyl/multibody/costs/centroidal-momentum.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"
#include "crocoddyl/multibody/costs/frame-velocity.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include "crocoddyl/multibody/numdiff/cost.hpp"

#include "state.hpp"
#include "activation.hpp"

#ifndef CROCODDYL_COST_FACTORY_HPP_
#define CROCODDYL_COST_FACTORY_HPP_

namespace crocoddyl {
namespace unittest {

struct CostModelTypes {
  enum Type {
    CostModelState,
    CostModelControl,
    CostModelCoMPosition,
    // CostModelCentroidalMomentum,  // @todo Figure out the pinocchio callbacks.
    CostModelFramePlacement,
    CostModelFrameRotation,
    CostModelFrameTranslation,
    CostModelFrameVelocity,
    NbCostModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbCostModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
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

class CostModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef typename MathBase::Vector6s Vector6d;

  explicit CostModelFactory() {}
  ~CostModelFactory() {}

  boost::shared_ptr<crocoddyl::CostModelAbstract> create(CostModelTypes::Type cost_type,
                                                         StateModelTypes::Type state_type,
                                                         ActivationModelTypes::Type activation_type) {
    StateModelFactory state_factory;
    ActivationModelFactory activation_factory;
    boost::shared_ptr<crocoddyl::CostModelAbstract> cost;
    boost::shared_ptr<crocoddyl::StateMultibody> state =
        boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(state_type));
    crocoddyl::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
    pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();

    switch (cost_type) {
      case CostModelTypes::CostModelState:
        cost = boost::make_shared<crocoddyl::CostModelState>(
            state, activation_factory.create(activation_type, state->get_ndx()), state->rand());
        break;
      case CostModelTypes::CostModelControl:
        cost = boost::make_shared<crocoddyl::CostModelControl>(
            state, activation_factory.create(activation_type, state->get_nv()),
            Eigen::VectorXd::Random(state->get_nv()));
        break;
      case CostModelTypes::CostModelCoMPosition:
        cost = boost::make_shared<crocoddyl::CostModelCoMPosition>(
            state, activation_factory.create(activation_type, 3), Eigen::Vector3d::Random());
        break;
      // case CostModelTypes::CostModelCentroidalMomentum:
      //   cost = boost::make_shared<crocoddyl::CostModelCentroidalMomentum>(state_,
      //                                                                      activation_factory.create(activation_type,
      //                                                                      6), Vector6d::Random());
      //   break;
      case CostModelTypes::CostModelFramePlacement:
        cost = boost::make_shared<crocoddyl::CostModelFramePlacement>(
            state, activation_factory.create(activation_type, 6), crocoddyl::FramePlacement(frame_index, frame_SE3));
        break;
      case CostModelTypes::CostModelFrameRotation:
        cost = boost::make_shared<crocoddyl::CostModelFrameRotation>(
            state, activation_factory.create(activation_type, 3),
            crocoddyl::FrameRotation(frame_index, frame_SE3.rotation()));
        break;
      case CostModelTypes::CostModelFrameTranslation:
        cost = boost::make_shared<crocoddyl::CostModelFrameTranslation>(
            state, activation_factory.create(activation_type, 3),
            crocoddyl::FrameTranslation(frame_index, frame_SE3.translation()));
        break;
      case CostModelTypes::CostModelFrameVelocity:
        cost = boost::make_shared<crocoddyl::CostModelFrameVelocity>(
            state, activation_factory.create(activation_type, 6),
            crocoddyl::FrameMotion(frame_index, pinocchio::Motion::Random()));
        break;
      default:
        throw_pretty(__FILE__ ": Wrong CostModelTypes::Type given");
        break;
    }
    return cost;
  }
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_COST_FACTORY_HPP_
