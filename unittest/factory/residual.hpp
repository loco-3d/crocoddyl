///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_RESIDUAL_FACTORY_HPP_
#define CROCODDYL_RESIDUAL_FACTORY_HPP_

#include "state.hpp"
#include "residual.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/numdiff/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
namespace unittest {

struct ResidualModelTypes {
  enum Type {
    ResidualModelState,
    ResidualModelControl,
    ResidualModelCoMPosition,
    ResidualModelCentroidalMomentum,
    ResidualModelFramePlacement,
    ResidualModelFrameRotation,
    ResidualModelFrameTranslation,
    ResidualModelFrameVelocity,
    ResidualModelControlGrav,
    NbResidualModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    for (int i = 0; i < NbResidualModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ResidualModelTypes::Type type);

class ResidualModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef typename MathBase::Vector6s Vector6d;

  explicit ResidualModelFactory();
  ~ResidualModelFactory();

  boost::shared_ptr<crocoddyl::ResidualModelAbstract> create(
      ResidualModelTypes::Type residual_type, StateModelTypes::Type state_type,
      std::size_t nu = std::numeric_limits<std::size_t>::max()) const;
};

boost::shared_ptr<crocoddyl::ResidualModelAbstract> create_random_residual(StateModelTypes::Type state_type);

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_RESIDUAL_FACTORY_HPP_
