///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2022, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "constraint.hpp"

#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-rotation.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ConstraintModelTypes::Type> ConstraintModelTypes::all(
    ConstraintModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ConstraintModelTypes::Type type) {
  switch (type) {
    case ConstraintModelTypes::ConstraintModelResidualStateEquality:
      os << "ConstraintModelResidualStateEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualStateInequality:
      os << "ConstraintModelResidualStateInequality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualControlEquality:
      os << "ConstraintModelResidualControlEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualControlInequality:
      os << "ConstraintModelResidualControlInequality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualCoMPositionEquality:
      os << "ConstraintModelResidualCoMPositionEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualCoMPositionInequality:
      os << "ConstraintModelResidualCoMPositionInequality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFramePlacementEquality:
      os << "ConstraintModelResidualFramePlacementEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFramePlacementInequality:
      os << "ConstraintModelResidualFramePlacementInequality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameRotationEquality:
      os << "ConstraintModelResidualFrameRotationEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameRotationInequality:
      os << "ConstraintModelResidualFrameRotationInequality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameTranslationEquality:
      os << "ConstraintModelResidualFrameTranslationEquality";
      break;
    case ConstraintModelTypes::
        ConstraintModelResidualFrameTranslationInequality:
      os << "ConstraintModelResidualFrameTranslationInequality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameVelocityEquality:
      os << "ConstraintModelResidualFrameVelocityEquality";
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameVelocityInequality:
      os << "ConstraintModelResidualFrameVelocityInequality";
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

std::shared_ptr<crocoddyl::ConstraintModelAbstract>
ConstraintModelFactory::create(ConstraintModelTypes::Type constraint_type,
                               StateModelTypes::Type state_type,
                               std::size_t nu) const {
  StateModelFactory state_factory;
  std::shared_ptr<crocoddyl::ConstraintModelAbstract> constraint;
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          state_factory.create(state_type));
  pinocchio::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();
  pinocchio::Motion frame_motion = pinocchio::Motion::Random();
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  Eigen::VectorXd lb, ub;
  switch (constraint_type) {
    case ConstraintModelTypes::ConstraintModelResidualStateEquality:
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelState>(
                     state, state->rand(), nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualStateInequality:
      lb = Eigen::VectorXd::Zero(state->get_ndx());
      ub = Eigen::VectorXd::Zero(state->get_ndx());
      state->diff(state->zero(), state->zero(), lb);
      state->diff(state->zero(), state->rand().cwiseAbs(), ub);
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state,
          std::make_shared<crocoddyl::ResidualModelState>(state, state->rand(),
                                                          nu),
          lb, ub.cwiseAbs());
      break;
    case ConstraintModelTypes::ConstraintModelResidualControlEquality:
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelControl>(
                     state, Eigen::VectorXd::Random(nu)));
      break;
    case ConstraintModelTypes::ConstraintModelResidualControlInequality:
      lb = Eigen::VectorXd::Zero(nu);
      lb(0) = -INFINITY;
      ub = Eigen::VectorXd::Random(nu).cwiseAbs();
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state,
          std::make_shared<crocoddyl::ResidualModelControl>(
              state, Eigen::VectorXd::Random(nu)),
          lb, ub);
      break;
    case ConstraintModelTypes::ConstraintModelResidualCoMPositionEquality:
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelCoMPosition>(
                     state, Eigen::Vector3d::Random(), nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualCoMPositionInequality:
      lb = Eigen::Vector3d(0., -INFINITY, 0.);
      ub = Eigen::Vector3d::Random().cwiseAbs();
      ub(2) = INFINITY;
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state,
          std::make_shared<crocoddyl::ResidualModelCoMPosition>(
              state, Eigen::Vector3d::Random(), nu),
          lb, ub);
      break;
    case ConstraintModelTypes::ConstraintModelResidualFramePlacementEquality:
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelFramePlacement>(
                     state, frame_index, frame_SE3, nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualFramePlacementInequality:
      lb = MathBaseTpl<double>::Vector6s::Zero();
      lb.tail<3>() << -INFINITY, -INFINITY, -INFINITY;
      ub = MathBaseTpl<double>::Vector6s::Random().cwiseAbs();
      ub.head<3>() << INFINITY, INFINITY, INFINITY;
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state,
          std::make_shared<crocoddyl::ResidualModelFramePlacement>(
              state, frame_index, frame_SE3, nu),
          lb, ub);
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameRotationEquality:
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelFrameRotation>(
                     state, frame_index, frame_SE3.rotation(), nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameRotationInequality:
      lb = Eigen::Vector3d::Zero();
      lb(1) = -INFINITY;
      ub = Eigen::Vector3d::Random().cwiseAbs();
      ub(2) = INFINITY;
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state,
          std::make_shared<crocoddyl::ResidualModelFrameRotation>(
              state, frame_index, frame_SE3.rotation(), nu),
          lb, ub);
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameTranslationEquality:
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelFrameTranslation>(
                     state, frame_index, frame_SE3.translation(), nu));
      break;
    case ConstraintModelTypes::
        ConstraintModelResidualFrameTranslationInequality:
      lb = -1 * Eigen::Vector3d::Random().cwiseAbs();
      ub = Eigen::Vector3d::Random().cwiseAbs();
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state,
          std::make_shared<crocoddyl::ResidualModelFrameTranslation>(
              state, frame_index, frame_SE3.translation(), nu),
          lb, ub);
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameVelocityEquality:
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelFrameVelocity>(
                     state, frame_index, frame_motion, pinocchio::LOCAL, nu));
      break;
    case ConstraintModelTypes::ConstraintModelResidualFrameVelocityInequality:
      lb = -1 * MathBaseTpl<double>::Vector6s::Random().cwiseAbs();
      lb(0) = -INFINITY;
      ub = MathBaseTpl<double>::Vector6s::Random().cwiseAbs();
      ub(0) = INFINITY;
      constraint = std::make_shared<crocoddyl::ConstraintModelResidual>(
          state,
          std::make_shared<crocoddyl::ResidualModelFrameVelocity>(
              state, frame_index, frame_motion, pinocchio::LOCAL, nu),
          lb, ub);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ConstraintModelType::Type given");
      break;
  }
  return constraint;
}

std::shared_ptr<crocoddyl::ConstraintModelAbstract> create_random_constraint(
    StateModelTypes::Type state_type) {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }

  ConstraintModelFactory factory;
  ConstraintModelTypes::Type rand_type =
      static_cast<ConstraintModelTypes::Type>(
          rand() % ConstraintModelTypes::NbConstraintModelTypes);
  return factory.create(rand_type, state_type);
}

}  // namespace unittest
}  // namespace crocoddyl
