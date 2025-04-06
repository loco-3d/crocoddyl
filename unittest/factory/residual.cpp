///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "residual.hpp"

#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/control-gravity.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-rotation.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#ifdef CROCODDYL_WITH_PAIR_COLLISION
#include "crocoddyl/multibody/residuals/pair-collision.hpp"
#endif  // CROCODDYL_WITH_PAIR_COLLISION
#include "crocoddyl/multibody/residuals/state.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ResidualModelTypes::Type> ResidualModelTypes::all(
    ResidualModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ResidualModelTypes::Type type) {
  switch (type) {
    case ResidualModelTypes::ResidualModelState:
      os << "ResidualModelState";
      break;
    case ResidualModelTypes::ResidualModelControl:
      os << "ResidualModelControl";
      break;
    case ResidualModelTypes::ResidualModelCoMPosition:
      os << "ResidualModelCoMPosition";
      break;
    case ResidualModelTypes::ResidualModelCentroidalMomentum:
      os << "ResidualModelCentroidalMomentum";
      break;
    case ResidualModelTypes::ResidualModelFramePlacement:
      os << "ResidualModelFramePlacement";
      break;
    case ResidualModelTypes::ResidualModelFrameRotation:
      os << "ResidualModelFrameRotation";
      break;
    case ResidualModelTypes::ResidualModelFrameTranslation:
      os << "ResidualModelFrameTranslation";
      break;
    case ResidualModelTypes::ResidualModelFrameVelocity:
      os << "ResidualModelFrameVelocity";
      break;
    case ResidualModelTypes::ResidualModelControlGrav:
      os << "ResidualModelControlGrav";
      break;
#ifdef PINOCCHIO_WITH_HPP_FCL
#ifdef CROCODDYL_WITH_PAIR_COLLISION
    case ResidualModelTypes::ResidualModelPairCollision:
      os << "ResidualModelPairCollision";
      break;
#endif  // CROCODDYL_WITH_PAIR_COLLISION
#endif  // PINOCCHIO_WITH_HPP_FCL
    case ResidualModelTypes::NbResidualModelTypes:
      os << "NbResidualModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ResidualModelFactory::ResidualModelFactory() {}
ResidualModelFactory::~ResidualModelFactory() {}

std::shared_ptr<crocoddyl::ResidualModelAbstract> ResidualModelFactory::create(
    ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type,
    std::size_t nu) const {
  StateModelFactory state_factory;
  std::shared_ptr<crocoddyl::ResidualModelAbstract> residual;
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          state_factory.create(state_type));
  pinocchio::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();

#ifdef PINOCCHIO_WITH_HPP_FCL
#ifdef CROCODDYL_WITH_PAIR_COLLISION
  pinocchio::SE3 frame_SE3_obstacle = pinocchio::SE3::Random();
  std::shared_ptr<pinocchio::GeometryModel> geometry =
      std::make_shared<pinocchio::GeometryModel>(pinocchio::GeometryModel());
  pinocchio::GeomIndex ig_frame =
      geometry->addGeometryObject(pinocchio::GeometryObject(
          "frame", frame_index,
          state->get_pinocchio()->frames[frame_index].parentJoint,
          std::make_shared<hpp::fcl::Sphere>(0), frame_SE3));
  pinocchio::GeomIndex ig_obs =
      geometry->addGeometryObject(pinocchio::GeometryObject(
          "obs", state->get_pinocchio()->getFrameId("universe"),
          state->get_pinocchio()
              ->frames[state->get_pinocchio()->getFrameId("universe")]
              .parentJoint,
          std::make_shared<hpp::fcl::Sphere>(0), frame_SE3_obstacle));
  geometry->addCollisionPair(pinocchio::CollisionPair(ig_frame, ig_obs));
#endif  // CROCODDYL_WITH_PAIR_COLLISION
#endif  // PINOCCHIO_WITH_HPP_FCL
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (residual_type) {
    case ResidualModelTypes::ResidualModelState:
      residual = std::make_shared<crocoddyl::ResidualModelState>(
          state, state->rand(), nu);
      break;
    case ResidualModelTypes::ResidualModelControl:
      residual = std::make_shared<crocoddyl::ResidualModelControl>(
          state, Eigen::VectorXd::Random(nu));
      break;
    case ResidualModelTypes::ResidualModelCoMPosition:
      residual = std::make_shared<crocoddyl::ResidualModelCoMPosition>(
          state, Eigen::Vector3d::Random(), nu);
      break;
    case ResidualModelTypes::ResidualModelCentroidalMomentum:
      residual = std::make_shared<crocoddyl::ResidualModelCentroidalMomentum>(
          state, Vector6d::Random(), nu);
      break;
    case ResidualModelTypes::ResidualModelFramePlacement:
      residual = std::make_shared<crocoddyl::ResidualModelFramePlacement>(
          state, frame_index, frame_SE3, nu);
      break;
    case ResidualModelTypes::ResidualModelFrameRotation:
      residual = std::make_shared<crocoddyl::ResidualModelFrameRotation>(
          state, frame_index, frame_SE3.rotation(), nu);
      break;
    case ResidualModelTypes::ResidualModelFrameTranslation:
      residual = std::make_shared<crocoddyl::ResidualModelFrameTranslation>(
          state, frame_index, frame_SE3.translation(), nu);
      break;
    case ResidualModelTypes::ResidualModelFrameVelocity:
      residual = std::make_shared<crocoddyl::ResidualModelFrameVelocity>(
          state, frame_index, pinocchio::Motion::Random(),
          static_cast<pinocchio::ReferenceFrame>(rand() % 2),
          nu);  // the code cannot test LOCAL_WORLD_ALIGNED
      break;
    case ResidualModelTypes::ResidualModelControlGrav:
      residual =
          std::make_shared<crocoddyl::ResidualModelControlGrav>(state, nu);
      break;
#ifdef PINOCCHIO_WITH_HPP_FCL
#ifdef CROCODDYL_WITH_PAIR_COLLISION
    case ResidualModelTypes::ResidualModelPairCollision:
      residual = std::make_shared<crocoddyl::ResidualModelPairCollision>(
          state, nu, geometry, 0,
          state->get_pinocchio()->frames[frame_index].parentJoint);
      break;
#endif  // CROCODDYL_WITH_PAIR_COLLISION
#endif  // PINOCCHIO_WITH_HPP_FCL
    default:
      throw_pretty(__FILE__ ": Wrong ResidualModelTypes::Type given");
      break;
  }
  return residual;
}

std::shared_ptr<crocoddyl::ResidualModelAbstract> create_random_residual(
    StateModelTypes::Type state_type) {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }

  ResidualModelFactory factory;
  ResidualModelTypes::Type rand_type = static_cast<ResidualModelTypes::Type>(
      rand() % ResidualModelTypes::NbResidualModelTypes);
  return factory.create(rand_type, state_type);
}

}  // namespace unittest
}  // namespace crocoddyl
