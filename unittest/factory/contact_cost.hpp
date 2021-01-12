///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACT_COST_FACTORY_HPP_
#define CROCODDYL_CONTACT_COST_FACTORY_HPP_

#include "state.hpp"
#include "actuation.hpp"
#include "activation.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"

namespace crocoddyl {
namespace unittest {

struct ContactCostModelTypes {
  enum Type {
    CostModelContactForce,
    CostModelContactCoPPosition,
    CostModelContactFrictionCone,
    CostModelContactWrenchCone,
    NbContactCostModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbContactCostModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ContactCostModelTypes::Type type);

class ContactCostModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;

  explicit ContactCostModelFactory();
  ~ContactCostModelFactory();

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create(ContactCostModelTypes::Type cost_type,
                                                                       PinocchioModelTypes::Type model_type,
                                                                       ActivationModelTypes::Type activation_type,
                                                                       ActuationModelTypes::Type actuation_type) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_CONTACT_COST_FACTORY_HPP_
