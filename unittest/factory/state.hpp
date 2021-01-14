///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_STATE_FACTORY_HPP_
#define CROCODDYL_STATE_FACTORY_HPP_

#include "pinocchio_model.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/numdiff/state.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

struct StateModelTypes {
  enum Type {
    StateVector,
    StateMultibody_TalosArm,
    StateMultibody_HyQ,
    StateMultibody_Talos,
    StateMultibody_RandomHumanoid,
    NbStateModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbStateModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, StateModelTypes::Type type);

class StateModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit StateModelFactory();
  ~StateModelFactory();

  boost::shared_ptr<crocoddyl::StateAbstract> create(StateModelTypes::Type state_type) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_STATE_FACTORY_HPP_
