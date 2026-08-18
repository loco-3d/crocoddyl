///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_DYNAMICS_FACTORY_HPP_
#define CROCODDYL_DYNAMICS_FACTORY_HPP_

#include "actuation.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/dynamics-base.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct DynamicsModelTypes {
  enum Type {
    DynamicsModelFreeFwd_Hector,
    DynamicsModelFreeFwd_TalosArm,
    DynamicsModelFreeInv_Hector,
    DynamicsModelFreeInv_TalosArm,
    DynamicsModelContactFwd_TalosArm,
    DynamicsModelContactFwd_HyQ,
    DynamicsModelContactFwd_Talos,
    DynamicsModelContactFwdWithFriction_HyQ,
    DynamicsModelContactFwdWithFriction_Talos,
    DynamicsModelContactInv_TalosArm,
    DynamicsModelContactInv_HyQ,
    DynamicsModelContactInv_Talos,
    DynamicsModelContactInvWithFriction_HyQ,
    DynamicsModelContactInvWithFriction_Talos,
    NbDynamicsModelTypes
  };

  static std::vector<Type> init_all() {
    std::vector<Type> types;
    types.reserve(NbDynamicsModelTypes);
    for (int i = 0; i < NbDynamicsModelTypes; ++i) {
      types.push_back(static_cast<Type>(i));
    }
    return types;
  }

  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, DynamicsModelTypes::Type type);

/**
 * @brief Components needed to construct a dynamics-backed action model
 */
struct DynamicsModelFactoryResult {
  std::shared_ptr<crocoddyl::DynamicsModelAbstract> dynamics;
  std::shared_ptr<crocoddyl::CostModelSum> costs;
  std::shared_ptr<crocoddyl::ConstraintModelManager> constraints;
};

/**
 * @brief Create representative multibody dynamics compositions for unit tests
 */
class DynamicsModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit DynamicsModelFactory();
  ~DynamicsModelFactory();

  DynamicsModelFactoryResult create(DynamicsModelTypes::Type type,
                                    const bool with_baumgarte = true) const;

  DynamicsModelFactoryResult create_freeFwdDynamics(
      StateModelTypes::Type state_type,
      ActuationModelTypes::Type actuation_type,
      const bool with_action_constraints = true) const;

  DynamicsModelFactoryResult create_freeInvDynamics(
      StateModelTypes::Type state_type,
      ActuationModelTypes::Type actuation_type,
      const bool with_action_constraints = true) const;

  DynamicsModelFactoryResult create_contactFwdDynamics(
      StateModelTypes::Type state_type,
      ActuationModelTypes::Type actuation_type, const bool with_friction = true,
      const bool with_baumgarte = true) const;

  DynamicsModelFactoryResult create_contactInvDynamics(
      StateModelTypes::Type state_type,
      ActuationModelTypes::Type actuation_type, const bool with_friction = true,
      const bool with_baumgarte = true) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_DYNAMICS_FACTORY_HPP_
