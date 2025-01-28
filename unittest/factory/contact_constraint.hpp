///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACT_CONSTRAINT_FACTORY_HPP_
#define CROCODDYL_CONTACT_CONSTRAINT_FACTORY_HPP_

#include "activation.hpp"
#include "actuation.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct ContactConstraintModelTypes {
  enum Type {
    ConstraintModelResidualContactForceEquality,
    ConstraintModelResidualContactCoPPositionInequality,
    ConstraintModelResidualContactFrictionConeInequality,
    ConstraintModelResidualContactWrenchConeInequality,
    ConstraintModelResidualContactControlGravInequality,
    NbContactConstraintModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbContactConstraintModelTypes);
    for (int i = 0; i < NbContactConstraintModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream &operator<<(std::ostream &os,
                         ContactConstraintModelTypes::Type type);

class ContactConstraintModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;

  explicit ContactConstraintModelFactory();
  ~ContactConstraintModelFactory();

  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create(
      ContactConstraintModelTypes::Type constraint_type,
      PinocchioModelTypes::Type model_type,
      ActuationModelTypes::Type actuation_type) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_CONTACT_CONSTRAINT_FACTORY_HPP_
