///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "residual.hpp"
// #include "crocoddyl/multibody/residuals/state.hpp"
// #include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
// // #include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
// #include "crocoddyl/multibody/residuals/frame-rotation.hpp"
// #include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
// #include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
// #include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ResidualModelTypes::Type> ResidualModelTypes::all(ResidualModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ResidualModelTypes::Type type) {
  switch (type) {
    // case ResidualModelTypes::ResidualModelState:
    //   os << "ResidualModelState";
    //   break;
    // case ResidualModelTypes::ResidualModelControl:
    //   os << "ResidualModelControl";
    //   break;
    case ResidualModelTypes::ResidualModelCoMPosition:
      os << "ResidualModelCoMPosition";
      break;
    // // case ResidualModelTypes::ResidualModelCentroidalMomentum:
    // //   os << "ResidualModelCentroidalMomentum";
    // //   break;
    case ResidualModelTypes::ResidualModelFramePlacement:
      os << "ResidualModelFramePlacement";
      break;
    // case ResidualModelTypes::ResidualModelFrameRotation:
    //   os << "ResidualModelFrameRotation";
    //   break;
    // case ResidualModelTypes::ResidualModelFrameTranslation:
    //   os << "ResidualModelFrameTranslation";
    //   break;
    case ResidualModelTypes::ResidualModelFrameVelocity:
      os << "ResidualModelFrameVelocity";
      break;
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

boost::shared_ptr<crocoddyl::ResidualModelAbstract> ResidualModelFactory::create(
    ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type, std::size_t nu) const {
  StateModelFactory state_factory;
  boost::shared_ptr<crocoddyl::ResidualModelAbstract> residual;
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(state_type));
  pinocchio::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (residual_type) {
    // case ResidualModelTypes::ResidualModelState:
    //   residual = boost::make_shared<crocoddyl::ResidualModelState>(
    //       state, activation_factory.create(activation_type, state->get_ndx()), state->rand(), nu);
    //   break;
    // case ResidualModelTypes::ResidualModelControl:
    //   residual = boost::make_shared<crocoddyl::ResidualModelControl>(state,
    //   activation_factory.create(activation_type, nu),
    //                                                          Eigen::VectorXd::Random(nu));
    //   break;
    case ResidualModelTypes::ResidualModelCoMPosition:
      residual = boost::make_shared<crocoddyl::ResidualModelCoMPosition>(state, Eigen::Vector3d::Random(), nu);
      break;
    // // case ResidualModelTypes::ResidualModelCentroidalMomentum:
    // //   residual = boost::make_shared<crocoddyl::ResidualModelCentroidalMomentum>(state_, Vector6d::Random(), nu);
    // //   break;
    case ResidualModelTypes::ResidualModelFramePlacement:
      residual = boost::make_shared<crocoddyl::ResidualModelFramePlacement>(state, frame_index, frame_SE3, nu);
      break;
    // case ResidualModelTypes::ResidualModelFrameRotation:
    //   residual = boost::make_shared<crocoddyl::ResidualModelFrameRotation>(
    //       state, activation_factory.create(activation_type, 3),
    //       crocoddyl::FrameRotation(frame_index, frame_SE3.rotation()), nu);
    //   break;
    // case ResidualModelTypes::ResidualModelFrameTranslation:
    //   residual = boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
    //       state, activation_factory.create(activation_type, 3),
    //       crocoddyl::FrameTranslation(frame_index, frame_SE3.translation()), nu);
    //   break;
    case ResidualModelTypes::ResidualModelFrameVelocity:
      residual = boost::make_shared<crocoddyl::ResidualModelFrameVelocity>(
          state, frame_index, pinocchio::Motion::Random(), static_cast<pinocchio::ReferenceFrame>(rand() % 2),
          nu);  // the code cannot test LOCAL_WORLD_ALIGNED
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ResidualModelTypes::Type given");
      break;
  }
  return residual;
}

boost::shared_ptr<crocoddyl::ResidualModelAbstract> create_random_residual(StateModelTypes::Type state_type) {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }

  ResidualModelFactory factory;
  ResidualModelTypes::Type rand_type =
      static_cast<ResidualModelTypes::Type>(rand() % ResidualModelTypes::NbResidualModelTypes);
  return factory.create(rand_type, state_type);
}

}  // namespace unittest
}  // namespace crocoddyl
