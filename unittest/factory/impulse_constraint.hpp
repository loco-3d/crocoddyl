///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_IMPULSE_CONSTRAINT_FACTORY_HPP_
#define CROCODDYL_IMPULSE_CONSTRAINT_FACTORY_HPP_

#include "activation.hpp"
#include "actuation.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct ImpulseConstraintModelTypes {
  enum Type {
    CostModelResidualImpulseCoMEquality,
    ConstraintModelResidualImpulseForceEquality,
    ConstraintModelResidualImpulseCoPPositionInequality,
    ConstraintModelResidualImpulseFrictionConeInequality,
    ConstraintModelResidualImpulseWrenchConeInequality,
    NbImpulseConstraintModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbImpulseConstraintModelTypes);
    for (int i = 0; i < NbImpulseConstraintModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream &operator<<(std::ostream &os,
                         ImpulseConstraintModelTypes::Type type);

class ImpulseConstraintModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;

  explicit ImpulseConstraintModelFactory();
  ~ImpulseConstraintModelFactory();

  std::shared_ptr<crocoddyl::ActionModelAbstract> create(
      ImpulseConstraintModelTypes::Type constraint_type,
      PinocchioModelTypes::Type model_type) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_IMPULSE_CONSTRAINT_FACTORY_HPP_
