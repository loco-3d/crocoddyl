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

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {
namespace unittest {

struct IntegratorTypes {
  enum Type { IntegratorEuler, IntegratorRK4, NbIntegratorTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
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

  boost::shared_ptr<crocoddyl::ActionModelAbstract> create(
      IntegratorTypes::Type type, boost::shared_ptr<DifferentialActionModelAbstract> model) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_INTEGRATOR_FACTORY_HPP_
