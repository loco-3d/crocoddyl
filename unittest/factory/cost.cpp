///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "cost.hpp"

#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/control-gravity.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
// #include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-rotation.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#ifdef CROCODDYL_WITH_PAIR_COLLISION
#include "crocoddyl/multibody/residuals/pair-collision.hpp"
#endif  // CROCODDYL_WITH_PAIR_COLLISION

namespace crocoddyl {
namespace unittest {

const std::vector<CostModelTypes::Type> CostModelTypes::all(
    CostModelTypes::init_all());
const std::vector<CostModelNoFFTypes::Type> CostModelNoFFTypes::all(
    CostModelNoFFTypes::init_all());
#ifdef PINOCCHIO_WITH_HPP_FCL
#ifdef CROCODDYL_WITH_PAIR_COLLISION
const std::vector<CostModelCollisionTypes::Type> CostModelCollisionTypes::all(
    CostModelCollisionTypes::init_all());
#endif  // CROCODDYL_WITH_PAIR_COLLISION
#endif  // PINOCCHIO_WITH_HPP_FCL

std::ostream& operator<<(std::ostream& os, CostModelTypes::Type type) {
  switch (type) {
    case CostModelTypes::CostModelResidualState:
      os << "CostModelResidualState";
      break;
    case CostModelTypes::CostModelResidualControl:
      os << "CostModelResidualControl";
      break;
    case CostModelTypes::CostModelResidualCoMPosition:
      os << "CostModelResidualCoMPosition";
      break;
    // case CostModelTypes::CostModelResidualCentroidalMomentum:
    //   os << "CostModelResidualCentroidalMomentum";
    //   break;
    case CostModelTypes::CostModelResidualFramePlacement:
      os << "CostModelResidualFramePlacement";
      break;
    case CostModelTypes::CostModelResidualFrameRotation:
      os << "CostModelResidualFrameRotation";
      break;
    case CostModelTypes::CostModelResidualFrameTranslation:
      os << "CostModelResidualFrameTranslation";
      break;
    case CostModelTypes::CostModelResidualFrameVelocity:
      os << "CostModelResidualFrameVelocity";
      break;
    case CostModelTypes::NbCostModelTypes:
      os << "NbCostModelTypes";
      break;
    default:
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, CostModelNoFFTypes::Type type) {
  switch (type) {
    case CostModelNoFFTypes::CostModelResidualControlGrav:
      os << "CostModelResidualControlGrav";
      break;
    case CostModelNoFFTypes::NbCostModelNoFFTypes:
      os << "NbCostModelNoFFTypes";
      break;
    default:
      break;
  }
  return os;
}

#ifdef PINOCCHIO_WITH_HPP_FCL
#ifdef CROCODDYL_WITH_PAIR_COLLISION
std::ostream& operator<<(std::ostream& os, CostModelCollisionTypes::Type type) {
  switch (type) {
    case CostModelCollisionTypes::CostModelResidualPairCollision:
      os << "CostModelResidualPairCollision";
      break;
    case CostModelCollisionTypes::NbCostModelCollisionTypes:
      os << "NbCostModelCollisionTypes";
      break;
    default:
      break;
  }
  return os;
}
#endif  // CROCODDYL_WITH_PAIR_COLLISION
#endif  // PINOCCHIO_WITH_HPP_FCL

CostModelFactory::CostModelFactory() {}
CostModelFactory::~CostModelFactory() {}

std::shared_ptr<crocoddyl::CostModelAbstract> CostModelFactory::create(
    CostModelTypes::Type cost_type, StateModelTypes::Type state_type,
    ActivationModelTypes::Type activation_type, std::size_t nu) const {
  StateModelFactory state_factory;
  ActivationModelFactory activation_factory;
  std::shared_ptr<crocoddyl::CostModelAbstract> cost;
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          state_factory.create(state_type));

  pinocchio::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();

  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (cost_type) {
    case CostModelTypes::CostModelResidualState:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, activation_factory.create(activation_type, state->get_ndx()),
          std::make_shared<crocoddyl::ResidualModelState>(state, state->zero(),
                                                          nu));
      break;
    case CostModelTypes::CostModelResidualControl:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, activation_factory.create(activation_type, nu),
          std::make_shared<crocoddyl::ResidualModelControl>(
              state, Eigen::VectorXd::Random(nu)));
      break;
    case CostModelTypes::CostModelResidualCoMPosition:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, activation_factory.create(activation_type, 3),
          std::make_shared<crocoddyl::ResidualModelCoMPosition>(
              state, Eigen::Vector3d::Random(), nu));
      break;
    // case CostModelTypes::CostModelResidualCentroidalMomentum:
    //   cost = std::make_shared<crocoddyl::CostModelResidual>(
    //       state,
    //       std::make_shared<crocoddyl::ResidualModelCentroidalMomentum>(state,
    //       Vector6d::Random(), nu), activation_factory.create(activation_type,
    //       6));
    //   break;
    case CostModelTypes::CostModelResidualFramePlacement:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, activation_factory.create(activation_type, 6),
          std::make_shared<crocoddyl::ResidualModelFramePlacement>(
              state, frame_index, frame_SE3, nu));
      break;
    case CostModelTypes::CostModelResidualFrameRotation:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, activation_factory.create(activation_type, 3),
          std::make_shared<crocoddyl::ResidualModelFrameRotation>(
              state, frame_index, frame_SE3.rotation(), nu));
      break;
    case CostModelTypes::CostModelResidualFrameTranslation:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, activation_factory.create(activation_type, 3),
          std::make_shared<crocoddyl::ResidualModelFrameTranslation>(
              state, frame_index, frame_SE3.translation(), nu));
      break;
    case CostModelTypes::CostModelResidualFrameVelocity:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, activation_factory.create(activation_type, 6),
          std::make_shared<crocoddyl::ResidualModelFrameVelocity>(
              state, frame_index, pinocchio::Motion::Random(),
              pinocchio::ReferenceFrame::LOCAL, nu));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong CostModelTypes::Type given");
      break;
  }
  return cost;
}

std::shared_ptr<crocoddyl::CostModelAbstract> CostModelFactory::create(
    CostModelNoFFTypes::Type cost_type,
    ActivationModelTypes::Type activation_type, std::size_t nu) const {
  StateModelFactory state_factory;
  ActivationModelFactory activation_factory;
  std::shared_ptr<crocoddyl::CostModelAbstract> cost;
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          state_factory.create(StateModelTypes::StateMultibody_TalosArm));
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }

  switch (cost_type) {
    case CostModelNoFFTypes::CostModelResidualControlGrav:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, activation_factory.create(activation_type, state->get_nv()),
          std::make_shared<ResidualModelControlGrav>(state, nu));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong CostModelTypes::Type given");
      break;
  }
  return cost;
}

#ifdef PINOCCHIO_WITH_HPP_FCL
#ifdef CROCODDYL_WITH_PAIR_COLLISION
std::shared_ptr<crocoddyl::CostModelAbstract> CostModelFactory::create(
    CostModelCollisionTypes::Type cost_type, StateModelTypes::Type state_type,
    std::size_t nu) const {
  StateModelFactory state_factory;
  std::shared_ptr<crocoddyl::CostModelAbstract> cost;
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          state_factory.create(state_type));
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  pinocchio::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();
  pinocchio::SE3 frame_SE3_obstacle = pinocchio::SE3::Random();
  double alpha = fabs(Eigen::VectorXd::Random(1)[0]);
  double beta = fabs(Eigen::VectorXd::Random(1)[0]);

  std::shared_ptr<pinocchio::GeometryModel> geometry =
      std::make_shared<pinocchio::GeometryModel>(pinocchio::GeometryModel());
  pinocchio::GeomIndex ig_frame =
      geometry->addGeometryObject(pinocchio::GeometryObject(
          "frame", frame_index,
          state->get_pinocchio()->frames[frame_index].parentJoint,
          CollisionGeometryPtr(new hpp::fcl::Capsule(0, alpha)), frame_SE3));
  pinocchio::GeomIndex ig_obs =
      geometry->addGeometryObject(pinocchio::GeometryObject(
          "obs", state->get_pinocchio()->getFrameId("universe"),
          state->get_pinocchio()
              ->frames[state->get_pinocchio()->getFrameId("universe")]
              .parentJoint,
          CollisionGeometryPtr(new hpp::fcl::Capsule(0, beta)),
          frame_SE3_obstacle));
  geometry->addCollisionPair(pinocchio::CollisionPair(ig_frame, ig_obs));

  switch (cost_type) {
    case CostModelCollisionTypes::CostModelResidualPairCollision:
      cost = std::make_shared<crocoddyl::CostModelResidual>(
          state, std::make_shared<crocoddyl::ActivationModelQuad>(3),
          std::make_shared<crocoddyl::ResidualModelPairCollision>(
              state, nu, geometry, 0,
              state->get_pinocchio()->frames[frame_index].parentJoint));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong CostModelTypes::Type given");
      break;
  }
  return cost;
}
#endif  // CROCODDYL_WITH_PAIR_COLLISION
#endif  // PINOCCHIO_WITH_HPP_FCL

std::shared_ptr<crocoddyl::CostModelAbstract> create_random_cost(
    StateModelTypes::Type state_type, std::size_t nu) {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }

  CostModelFactory factory;
  CostModelTypes::Type rand_type = static_cast<CostModelTypes::Type>(
      rand() % CostModelTypes::NbCostModelTypes);
  return factory.create(rand_type, state_type,
                        ActivationModelTypes::ActivationModelQuad, nu);
}

}  // namespace unittest
}  // namespace crocoddyl
