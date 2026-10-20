///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_DYNAMICS_PARAMETER_ACCESS_HXX_
#define CROCODDYL_CORE_INTEGRATOR_DYNAMICS_PARAMETER_ACCESS_HXX_

#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/dynamics/constrained-inverse.hpp"
#include "crocoddyl/multibody/dynamics/impulse-forward.hpp"

namespace crocoddyl {
namespace internal {

template <typename Scalar>
std::shared_ptr<ParameterManagerTpl<Scalar> > getDynamicsParameters(
    const std::shared_ptr<DynamicsModelAbstractTpl<Scalar> >& dynamics) {
  const std::shared_ptr<DynamicsModelConstrainedForwardTpl<Scalar> > forward =
      std::dynamic_pointer_cast<DynamicsModelConstrainedForwardTpl<Scalar> >(
          dynamics);
  if (forward != nullptr) {
    return forward->get_params();
  }
  const std::shared_ptr<DynamicsModelConstrainedInverseTpl<Scalar> > inverse =
      std::dynamic_pointer_cast<DynamicsModelConstrainedInverseTpl<Scalar> >(
          dynamics);
  if (inverse != nullptr) {
    return inverse->get_params();
  }
  const std::shared_ptr<DynamicsModelImpulseForwardTpl<Scalar> > impulse =
      std::dynamic_pointer_cast<DynamicsModelImpulseForwardTpl<Scalar> >(
          dynamics);
  if (impulse != nullptr) {
    return impulse->get_params();
  }
  return std::shared_ptr<ParameterManagerTpl<Scalar> >();
}

}  // namespace internal
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_INTEGRATOR_DYNAMICS_PARAMETER_ACCESS_HXX_
