///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, University of Edinburgh, CTU, INRIA,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_DIFF_ACTION_FACTORY_HPP_
#define CROCODDYL_DIFF_ACTION_FACTORY_HPP_

#include "actuation.hpp"
#include "constraint.hpp"
#include "contact.hpp"
#include "cost.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actions/contact-invdyn.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/actions/free-invdyn.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct DifferentialActionModelTypes {
  enum Type {
    DifferentialActionModelLQR,
    DifferentialActionModelLQRDriftFree,
    DifferentialActionModelRandomLQR,
    DifferentialActionModelFreeFwdDynamics_Hector,
    DifferentialActionModelFreeFwdDynamics_TalosArm,
    DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed,
    DifferentialActionModelFreeInvDynamics_Hector,
    DifferentialActionModelFreeInvDynamics_TalosArm,
    DifferentialActionModelFreeInvDynamics_TalosArm_Squashed,
    DifferentialActionModelContactFwdDynamics_TalosArm,
    DifferentialActionModelContact2DFwdDynamics_TalosArm,
    DifferentialActionModelContactFwdDynamics_HyQ,
    DifferentialActionModelContactFwdDynamics_Talos,
    DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm,
    DifferentialActionModelContact2DFwdDynamicsWithFriction_TalosArm,
    DifferentialActionModelContactFwdDynamicsWithFriction_HyQ,
    DifferentialActionModelContactFwdDynamicsWithFriction_Talos,
    DifferentialActionModelContactInvDynamics_TalosArm,
    DifferentialActionModelContactInvDynamics_HyQ,
    DifferentialActionModelContactInvDynamics_Talos,
    DifferentialActionModelContactInvDynamicsWithFriction_TalosArm,
    DifferentialActionModelContactInvDynamicsWithFriction_HyQ,
    DifferentialActionModelContactInvDynamicsWithFriction_Talos,
    NbDifferentialActionModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbDifferentialActionModelTypes);
    for (int i = 0; i < NbDifferentialActionModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os,
                         DifferentialActionModelTypes::Type type);

class DifferentialActionModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit DifferentialActionModelFactory();
  ~DifferentialActionModelFactory();

  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create(
      DifferentialActionModelTypes::Type type,
      bool with_baumgarte = true) const;

  std::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics>
  create_freeFwdDynamics(StateModelTypes::Type state_type,
                         ActuationModelTypes::Type actuation_type,
                         bool constraints = true) const;

  std::shared_ptr<crocoddyl::DifferentialActionModelFreeInvDynamics>
  create_freeInvDynamics(StateModelTypes::Type state_type,
                         ActuationModelTypes::Type actuation_type,
                         bool constraints = true) const;

  std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
  create_contactFwdDynamics(StateModelTypes::Type state_type,
                            ActuationModelTypes::Type actuation_type,
                            bool with_friction = true,
                            bool with_baumgarte = true) const;

  std::shared_ptr<crocoddyl::DifferentialActionModelContactInvDynamics>
  create_contactInvDynamics(StateModelTypes::Type state_type,
                            ActuationModelTypes::Type actuation_type,
                            bool with_friction = true,
                            bool with_baumgarte = true) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_DIFF_ACTION_FACTORY_HPP_
