///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2022, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONSTRAINT_FACTORY_HPP_
#define CROCODDYL_CONSTRAINT_FACTORY_HPP_

#include "crocoddyl/core/constraint-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/numdiff/constraint.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct ConstraintModelTypes {
  enum Type {
    ConstraintModelResidualStateEquality,
    ConstraintModelResidualStateInequality,
    ConstraintModelResidualControlEquality,
    ConstraintModelResidualControlInequality,
    ConstraintModelResidualCoMPositionEquality,
    ConstraintModelResidualCoMPositionInequality,
    ConstraintModelResidualFramePlacementEquality,
    ConstraintModelResidualFramePlacementInequality,
    ConstraintModelResidualFrameRotationEquality,
    ConstraintModelResidualFrameRotationInequality,
    ConstraintModelResidualFrameTranslationEquality,
    ConstraintModelResidualFrameTranslationInequality,
    ConstraintModelResidualFrameVelocityEquality,
    ConstraintModelResidualFrameVelocityInequality,
    NbConstraintModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbConstraintModelTypes);
    for (int i = 0; i < NbConstraintModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ConstraintModelTypes::Type type);

class ConstraintModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ConstraintModelFactory();
  ~ConstraintModelFactory();

  std::shared_ptr<crocoddyl::ConstraintModelAbstract> create(
      ConstraintModelTypes::Type constraint_type,
      StateModelTypes::Type state_type,
      std::size_t nu = std::numeric_limits<std::size_t>::max()) const;
};

std::shared_ptr<crocoddyl::ConstraintModelAbstract> create_random_constraint(
    StateModelTypes::Type state_type);

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_COST_FACTORY_HPP_
