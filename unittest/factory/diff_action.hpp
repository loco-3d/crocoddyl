///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ACTION_FACTORY_HPP_
#define CROCODDYL_ACTION_FACTORY_HPP_

#include "state.hpp"
#include "actuation.hpp"
#include "cost.hpp"
#include "contact.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"

namespace crocoddyl {
namespace unittest {

struct DifferentialActionModelTypes {
  enum Type {
    DifferentialActionModelLQR,
    DifferentialActionModelLQRDriftFree,
    DifferentialActionModelFreeFwdDynamics_TalosArm,
    DifferentialActionModelContactFwdDynamics_HyQ,
    DifferentialActionModelContactFwdDynamics_Talos,
    DifferentialActionModelContactFwdDynamicsWithFriction_HyQ,
    DifferentialActionModelContactFwdDynamicsWithFriction_Talos,
    NbDifferentialActionModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbDifferentialActionModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, DifferentialActionModelTypes::Type type);

class DifferentialActionModelFactory {
 public:
  explicit DifferentialActionModelFactory();
  ~DifferentialActionModelFactory();

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create(DifferentialActionModelTypes::Type type);

 private:
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create_freeFwdDynamics(
      StateModelTypes::Type state_type);

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create_contactFwdDynamics(
      StateModelTypes::Type state_type, bool with_friction = true);
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_ACTION_FACTORY_HPP_
