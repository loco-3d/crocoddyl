///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_TASK_FACTORY_HPP_
#define CROCODDYL_TASK_FACTORY_HPP_

#include <limits>
#include <memory>
#include <ostream>
#include <vector>

#include "crocoddyl/core/task-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct TaskModelTypes {
  enum Type {
    TaskModelFrameRotation,
    TaskModelFrameTranslation,
    TaskModelFramePlacement,
    TaskModelCoMPosition,
    TaskModelCentroidalMomentum,
    TaskModelJointPosition,
    NbTaskModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbTaskModelTypes);
    for (int i = 0; i < NbTaskModelTypes; ++i) {
      v.push_back(static_cast<Type>(i));
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, TaskModelTypes::Type type);

class TaskModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit TaskModelFactory();
  ~TaskModelFactory();

  std::shared_ptr<crocoddyl::TaskModelAbstract> create(
      TaskModelTypes::Type task_type, StateModelTypes::Type state_type,
      std::size_t nu = std::numeric_limits<std::size_t>::max()) const;
};

std::shared_ptr<crocoddyl::TaskModelAbstract> create_random_task(
    StateModelTypes::Type state_type);

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_TASK_FACTORY_HPP_
