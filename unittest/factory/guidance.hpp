///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_GUIDANCE_FACTORY_HPP_
#define CROCODDYL_GUIDANCE_FACTORY_HPP_

#include "crocoddyl/core/guidance-base.hpp"

namespace crocoddyl {
namespace unittest {

struct GuidanceModelTypes {
  enum Type {
    GuidanceModelLinear,
    GuidanceModelSmoothSaturation,
    GuidanceModelComponentwiseSaturation,
    NbGuidanceModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbGuidanceModelTypes);
    for (int i = 0; i < NbGuidanceModelTypes; ++i) {
      v.push_back(static_cast<Type>(i));
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, GuidanceModelTypes::Type type);

class GuidanceModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit GuidanceModelFactory();
  ~GuidanceModelFactory();

  std::shared_ptr<crocoddyl::GuidanceModelAbstract> create(
      GuidanceModelTypes::Type guidance_type, std::size_t nr = 5) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_GUIDANCE_FACTORY_HPP_
