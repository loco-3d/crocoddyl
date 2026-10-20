///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ACTION_FACTORY_HPP_
#define CROCODDYL_ACTION_FACTORY_HPP_

#include <iterator>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {
namespace unittest {

struct ActionModelTypes {
  enum Type {
    ActionModelUnicycle,
    ActionModelLQRDriftFree,
    ActionModelLQR,
    ActionModelRandomLQR,
    ActionModelRandomLQRwithTerminalConstraint,
    NbActionModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbActionModelTypes);
    for (int i = 0; i < NbActionModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;

  static bool hasTerminalConstraints(Type type) {
    return type == ActionModelRandomLQRwithTerminalConstraint;
  }
};

std::ostream& operator<<(std::ostream& os, ActionModelTypes::Type type);

class ActionModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ActionModelFactory();
  ~ActionModelFactory();

  enum Instance { First, Second, Terminal };

  std::shared_ptr<crocoddyl::ActionModelAbstract> create(
      ActionModelTypes::Type type, Instance instance = Instance::First) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_ACTION_FACTORY_HPP_
