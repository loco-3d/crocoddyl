///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activations/smooth-1norm.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

template <typename Scalar>
class ActivationModelSmoothAbsTpl : public ActivationModelSmooth1NormTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef ActivationModelSmooth1NormTpl<Scalar> Base;

  DEPRECATED("Use ActivationModelSmooth1Norm",
             explicit ActivationModelSmoothAbsTpl(const std::size_t nr, const Scalar eps = Scalar(1.))
             : Base(nr, eps){};)
};

template <typename Scalar>
struct ActivationDataSmoothAbsTpl : public ActivationDataSmooth1NormTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef ActivationDataSmooth1NormTpl<Scalar> Base;

  template <typename Activation>
  DEPRECATED("Use ActivationDataSmooth1Norm", explicit ActivationDataSmoothAbsTpl(Activation* const activation)
             : Base(activation){})
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_