///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_DYNAMICS_PARAMETER_CAST_HPP_
#define CROCODDYL_MULTIBODY_DYNAMICS_PARAMETER_CAST_HPP_

#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"

namespace crocoddyl {
namespace internal {

template <typename Scalar, typename NewScalar>
std::shared_ptr<ParameterManagerTpl<NewScalar> > castDynamicsParameters(
    const std::shared_ptr<ParameterManagerTpl<Scalar> >& params,
    const std::shared_ptr<StateMultibodyTpl<NewScalar> >& state,
    const std::shared_ptr<ActuationModelAbstractTpl<NewScalar> >& actuation) {
  typedef ParameterManagerTpl<NewScalar> ParameterManagerNew;
  typedef ActionModelParamsAbstractTpl<NewScalar> ActionParamsNew;
  typedef DynamicsParamsAbstractTpl<NewScalar> DynamicsParamsNew;
  typedef ActuationMultibodyParamsTpl<Scalar> ActuationParams;
  typedef ActuationMultibodyParamsTpl<NewScalar> ActuationParamsNew;
  typedef ActuationModelMultibodyTpl<NewScalar> ActuationModelNew;

  const std::shared_ptr<ParameterManagerNew> ret =
      std::make_shared<ParameterManagerNew>(state);
  typename ParameterManagerTpl<Scalar>::ParameterContainer::const_iterator it,
      end;
  for (it = params->get_action_params().begin(),
      end = params->get_action_params().end();
       it != end; ++it) {
    const std::shared_ptr<ActionParamsNew> casted_param =
        std::dynamic_pointer_cast<ActionParamsNew>(
            it->second->get_param()->template cast<NewScalar>());
    if (casted_param == nullptr) {
      throw_pretty("Invalid call: parameter '"
                   << it->first
                   << "' is not an action parameter after casting");
    }
    ret->addParam(it->first, casted_param, it->second->get_active());
  }
  for (it = params->get_dynamics_params().begin(),
      end = params->get_dynamics_params().end();
       it != end; ++it) {
    std::shared_ptr<DynamicsParamsNew> casted_param;
    const std::shared_ptr<ActuationParams> actuation_param =
        std::dynamic_pointer_cast<ActuationParams>(it->second->get_param());
    if (actuation_param != nullptr && actuation != nullptr) {
      const std::shared_ptr<ActuationModelNew> multibody_actuation =
          std::dynamic_pointer_cast<ActuationModelNew>(actuation);
      if (multibody_actuation == nullptr) {
        throw_pretty(
            "Invalid call: multibody actuation parameters require a "
            "multibody actuation model");
      }
      const ActuationParamsNew standalone =
          actuation_param->template cast<NewScalar>();
      const std::shared_ptr<ActuationParamsNew> coherent =
          std::make_shared<ActuationParamsNew>(multibody_actuation);
      coherent->set_lb(standalone.get_lb());
      coherent->set_ub(standalone.get_ub());
      casted_param = coherent;
    } else {
      casted_param = std::dynamic_pointer_cast<DynamicsParamsNew>(
          it->second->get_param()->template cast<NewScalar>());
    }
    if (casted_param == nullptr) {
      throw_pretty("Invalid call: parameter '"
                   << it->first
                   << "' is not a dynamics parameter after casting");
    }
    ret->addParam(it->first, casted_param, it->second->get_active());
  }
  return ret;
}

}  // namespace internal
}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_DYNAMICS_PARAMETER_CAST_HPP_
