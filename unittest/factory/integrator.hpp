///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_INTEGRATOR_FACTORY_HPP_
#define CROCODDYL_INTEGRATOR_FACTORY_HPP_

#include <iterator>

#include "control.hpp"
#include "crocoddyl/core/integ-action-base.hpp"

namespace crocoddyl {
namespace unittest {

struct IntegratorTypes {
  enum Type {
    IntegratorEuler,
    IntegratorRK2,
    IntegratorRK3,
    IntegratorRK4,
    NbIntegratorTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbIntegratorTypes);
    for (int i = 0; i < NbIntegratorTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, IntegratorTypes::Type type);

class IntegratorFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit IntegratorFactory();
  ~IntegratorFactory();

  std::shared_ptr<crocoddyl::IntegratedActionModelAbstract> create(
      IntegratorTypes::Type type,
      std::shared_ptr<DifferentialActionModelAbstract> model) const;

  std::shared_ptr<crocoddyl::IntegratedActionModelAbstract> create(
      IntegratorTypes::Type type,
      std::shared_ptr<DifferentialActionModelAbstract> model,
      std::shared_ptr<ControlParametrizationModelAbstract> control) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_INTEGRATOR_FACTORY_HPP_
