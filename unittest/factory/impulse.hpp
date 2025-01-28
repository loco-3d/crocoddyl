///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_IMPULSES_FACTORY_HPP_
#define CROCODDYL_IMPULSES_FACTORY_HPP_

#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct ImpulseModelTypes {
  enum Type {
    ImpulseModel3D_LOCAL,
    ImpulseModel3D_WORLD,
    ImpulseModel3D_LWA,
    ImpulseModel6D_LOCAL,
    ImpulseModel6D_WORLD,
    ImpulseModel6D_LWA,
    NbImpulseModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbImpulseModelTypes);
    for (int i = 0; i < NbImpulseModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, const ImpulseModelTypes::Type& type);

class ImpulseModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ImpulseModelFactory();
  ~ImpulseModelFactory();

  std::shared_ptr<crocoddyl::ImpulseModelAbstract> create(
      ImpulseModelTypes::Type impulse_type,
      PinocchioModelTypes::Type model_type,
      const std::string frame_name = std::string("")) const;
};

std::shared_ptr<crocoddyl::ImpulseModelAbstract> create_random_impulse();

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_IMPULSES_FACTORY_HPP_
