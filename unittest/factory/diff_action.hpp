///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS
// Copyright (C) 2020 CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_DIFF_ACTION_FACTORY_HPP_
#define CROCODDYL_DIFF_ACTION_FACTORY_HPP_

#include "state.hpp"
#include "actuation.hpp"
#include "cost.hpp"
#include "contact.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"

namespace crocoddyl {
namespace unittest {

struct DifferentialActionModelTypes {
  enum Type {
    DifferentialActionModelLQR,
    DifferentialActionModelLQRDriftFree,
    DifferentialActionModelFreeFwdDynamics_TalosArm,
    DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed,
    DifferentialActionModelContactFwdDynamics_TalosArm,
    DifferentialActionModelContactFwdDynamics_HyQ,
    DifferentialActionModelContactFwdDynamics_Talos,
    DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm,
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
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit DifferentialActionModelFactory();
  ~DifferentialActionModelFactory();

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create(DifferentialActionModelTypes::Type type) const;

 private:
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> create_freeFwdDynamics(
      StateModelTypes::Type state_type, ActuationModelTypes::Type actuation_type) const;

  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> create_contactFwdDynamics(
      StateModelTypes::Type state_type, ActuationModelTypes::Type actuation_type, bool with_friction = true) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_DIFF_ACTION_FACTORY_HPP_
