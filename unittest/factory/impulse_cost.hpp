///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_IMPULSE_COST_FACTORY_HPP_
#define CROCODDYL_IMPULSE_COST_FACTORY_HPP_

#include "state.hpp"
#include "activation.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {
namespace unittest {

struct ImpulseCostModelTypes {
  enum Type {
    CostModelResidualImpulseCoM,
    CostModelResidualContactForce,
    CostModelResidualContactCoPPosition,
    CostModelResidualContactFrictionCone,
    CostModelResidualContactWrenchCone,
    NbImpulseCostModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbImpulseCostModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ImpulseCostModelTypes::Type type);

class ImpulseCostModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef typename MathBase::Vector6s Vector6d;

  explicit ImpulseCostModelFactory();
  ~ImpulseCostModelFactory();

  boost::shared_ptr<crocoddyl::ActionModelAbstract> create(ImpulseCostModelTypes::Type cost_type,
                                                           PinocchioModelTypes::Type model_type,
                                                           ActivationModelTypes::Type activation_type) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_IMPULSE_COST_FACTORY_HPP_
