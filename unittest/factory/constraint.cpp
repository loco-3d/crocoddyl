///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "constraint.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-rotation.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ConstraintModelTypes::Type> ConstraintModelTypes::all(ConstraintModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ConstraintModelTypes::Type type) {
  switch (type) {
    case ConstraintModelTypes::ConstraintModelResidualStateEquality:
      os << "ConstraintModelResidualStateEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualControlEquality:
      os << "ConstraintModelResidualControlEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualCoMPositionEquality:
      os << "ConstraintModelResidualCoMPositionEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFramePlacementEquality:
      os << "ConstraintModelResidualFramePlacementEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameRotationEquality:
      os << "ConstraintModelResidualFrameRotationEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameTranslationEquality:
      os << "ConstraintModelResidualFrameTranslationEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameVelocityEquality:
      os << "ConstraintModelResidualFrameVelocityEquality";
      break;
    case ConstraintModelTypes::NbConstraintModelTypes:
      os << "NbConstraintModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ConstraintModelFactory::ConstraintModelFactory() {}
ConstraintModelFactory::~ConstraintModelFactory() {}

boost::shared_ptr<crocoddyl::ConstraintModelAbstract> ConstraintModelFactory::create(
    ConstraintModelTypes::Type constraint_type, StateModelTypes::Type state_type, std::size_t nu) const {
  StateModelFactory state_factory;
  boost::shared_ptr<crocoddyl::ConstraintModelAbstract> constraint;
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create(state_type));
  pinocchio::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();
  pinocchio::Motion frame_motion = pinocchio::Motion::Random();
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (constraint_type) {
    case ConstraintModelTypes::ConstraintModelResidualStateEquality:
      constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelState>(state, state->rand(), nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualControlEquality:
      constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelControl>(state, Eigen::VectorXd::Random(nu)));
      break;
    case ConstraintModelTypes::ConstraintModelResidualCoMPositionEquality:
      constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelCoMPosition>(state, Eigen::Vector3d::Random(), nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualFramePlacementEquality:
      constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelFramePlacement>(state, frame_index, frame_SE3, nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameRotationEquality:
      constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
          state,
          boost::make_shared<crocoddyl::ResidualModelFrameRotation>(state, frame_index, frame_SE3.rotation(), nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameTranslationEquality:
      constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(state, frame_index,
                                                                              frame_SE3.translation(), nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameVelocityEquality:
      constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelFrameVelocity>(state, frame_index, frame_motion,
                                                                           pinocchio::LOCAL, nu));
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ConstraintModelType::Type given");
      break;
  }
  return constraint;
}

boost::shared_ptr<crocoddyl::ConstraintModelAbstract> create_random_constraint(StateModelTypes::Type state_type) {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }

  ConstraintModelFactory factory;
  ConstraintModelTypes::Type rand_type =
      static_cast<ConstraintModelTypes::Type>(rand() % ConstraintModelTypes::NbConstraintModelTypes);
  return factory.create(rand_type, state_type);
}

}  // namespace unittest
}  // namespace crocoddyl
