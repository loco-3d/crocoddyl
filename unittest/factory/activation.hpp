///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ACTIVATION_FACTORY_HPP_
#define CROCODDYL_ACTIVATION_FACTORY_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/numdiff/activation.hpp"

namespace crocoddyl {
namespace unittest {

struct ActivationModelTypes {
  enum Type {
    ActivationModelQuad,
    ActivationModelQuadFlatExp,
    ActivationModelQuadFlatLog,
    ActivationModelSmooth1Norm,
    ActivationModelSmooth2Norm,
    ActivationModelWeightedQuad,
    ActivationModelQuadraticBarrier,
    ActivationModelWeightedQuadraticBarrier,
    ActivationModel2NormBarrier,
    NbActivationModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbActivationModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ActivationModelTypes::Type type);

class ActivationModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ActivationModelFactory();
  ~ActivationModelFactory();

  boost::shared_ptr<crocoddyl::ActivationModelAbstract> create(ActivationModelTypes::Type activation_type,
                                                               std::size_t nr = 5) const;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_ACTIVATION_FACTORY_HPP_
