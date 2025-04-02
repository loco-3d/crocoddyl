///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTROL_FACTORY_HPP_
#define CROCODDYL_CONTROL_FACTORY_HPP_

#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {
namespace unittest {

struct ControlTypes {
  enum Type { PolyZero, PolyOne, PolyTwoRK3, PolyTwoRK4, NbControlTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbControlTypes);
    for (int i = 0; i < NbControlTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ControlTypes::Type type);

class ControlFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ControlFactory();
  ~ControlFactory();

  std::shared_ptr<crocoddyl::ControlParametrizationModelAbstract> create(
      ControlTypes::Type control_type, const std::size_t nu) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_CONTROL_FACTORY_HPP_
