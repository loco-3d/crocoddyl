///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "task.hpp"

#include "../random_generator.hpp"
#include "crocoddyl/multibody/tasks/centroidal-momentum.hpp"
#include "crocoddyl/multibody/tasks/com-position.hpp"
#include "crocoddyl/multibody/tasks/frame-placement.hpp"
#include "crocoddyl/multibody/tasks/frame-rotation.hpp"
#include "crocoddyl/multibody/tasks/frame-translation.hpp"
#include "crocoddyl/multibody/tasks/joint-position.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<TaskModelTypes::Type> TaskModelTypes::all(
    TaskModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, TaskModelTypes::Type type) {
  switch (type) {
    case TaskModelTypes::TaskModelFrameRotation:
      os << "TaskModelFrameRotation";
      break;
    case TaskModelTypes::TaskModelFrameTranslation:
      os << "TaskModelFrameTranslation";
      break;
    case TaskModelTypes::TaskModelFramePlacement:
      os << "TaskModelFramePlacement";
      break;
    case TaskModelTypes::TaskModelCoMPosition:
      os << "TaskModelCoMPosition";
      break;
    case TaskModelTypes::TaskModelCentroidalMomentum:
      os << "TaskModelCentroidalMomentum";
      break;
    case TaskModelTypes::TaskModelJointPosition:
      os << "TaskModelJointPosition";
      break;
    case TaskModelTypes::NbTaskModelTypes:
      os << "NbTaskModelTypes";
      break;
    default:
      break;
  }
  return os;
}

TaskModelFactory::TaskModelFactory() {}
TaskModelFactory::~TaskModelFactory() {}

std::shared_ptr<crocoddyl::TaskModelAbstract> TaskModelFactory::create(
    TaskModelTypes::Type task_type, StateModelTypes::Type state_type,
    std::size_t nu) const {
  StateModelFactory state_factory;
  std::shared_ptr<crocoddyl::StateAbstract> state_base =
      state_factory.create(state_type);
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::dynamic_pointer_cast<crocoddyl::StateMultibody>(state_base);
  if (state == nullptr) {
    throw_pretty(__FILE__ ": TaskModelFactory requires a multibody state");
  }

  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }

  pinocchio::FrameIndex frame_index = state->get_pinocchio()->frames.size() - 1;
  pinocchio::SE3 frame_SE3 = pinocchio::SE3::Random();
  Eigen::Vector3d com_ref = Eigen::Vector3d::Random();
  Eigen::Matrix<double, 6, 1> h_ref = Eigen::Matrix<double, 6, 1>::Random();
  Eigen::Matrix<double, 6, 1> hdot_ref = Eigen::Matrix<double, 6, 1>::Random();
  Eigen::VectorXd x_ref = state->rand();
  Eigen::VectorXd a_ref = Eigen::VectorXd::Random(state->get_nv());
  std::shared_ptr<crocoddyl::TaskModelAbstract> task;

  switch (task_type) {
    case TaskModelTypes::TaskModelFrameRotation: {
      const auto type = static_cast<pinocchio::ReferenceFrame>(
          random_int_in_range<int>(0, 2));
      task = std::make_shared<crocoddyl::TaskModelFrameRotation>(
          state, frame_index, frame_SE3.rotation(), type, nu);
    } break;
    case TaskModelTypes::TaskModelFrameTranslation: {
      const auto type = static_cast<pinocchio::ReferenceFrame>(
          random_int_in_range<int>(0, 2));
      task = std::make_shared<crocoddyl::TaskModelFrameTranslation>(
          state, frame_index, frame_SE3.translation(), type, nu);
    } break;
    case TaskModelTypes::TaskModelFramePlacement: {
      const auto type = static_cast<pinocchio::ReferenceFrame>(
          random_int_in_range<int>(0, 2));
      task = std::make_shared<crocoddyl::TaskModelFramePlacement>(
          state, frame_index, frame_SE3, type, nu);
    } break;
    case TaskModelTypes::TaskModelCoMPosition:
      task =
          std::make_shared<crocoddyl::TaskModelCoMPosition>(state, com_ref, nu);
      break;
    case TaskModelTypes::TaskModelCentroidalMomentum:
      task = std::make_shared<crocoddyl::TaskModelCentroidalMomentum>(
          state, h_ref, hdot_ref, nu);
      break;
    case TaskModelTypes::TaskModelJointPosition:
      task = std::make_shared<crocoddyl::TaskModelJointPosition>(state, x_ref,
                                                                 a_ref, nu);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong TaskModelTypes::Type given");
      break;
  }
  return task;
}

std::shared_ptr<crocoddyl::TaskModelAbstract> create_random_task(
    StateModelTypes::Type state_type) {
  TaskModelFactory factory;
  TaskModelTypes::Type rand_type = static_cast<TaskModelTypes::Type>(
      random_int_in_range<int>(0, TaskModelTypes::NbTaskModelTypes - 1));
  return factory.create(rand_type, state_type);
}

}  // namespace unittest
}  // namespace crocoddyl
