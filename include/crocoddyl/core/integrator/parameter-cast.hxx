///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_PARAMETER_CAST_HXX_
#define CROCODDYL_CORE_INTEGRATOR_PARAMETER_CAST_HXX_

#include "crocoddyl/core/integrator/dynamics-parameter-access.hxx"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {
namespace detail {

template <typename Scalar, typename NewScalar>
std::shared_ptr<ParameterManagerTpl<NewScalar> > cast_integrated_action_params(
    const std::shared_ptr<ParameterManagerTpl<Scalar> >& params,
    const std::shared_ptr<IntegratorTimeTpl<Scalar> >& integrator_time,
    const std::shared_ptr<IntegratorTimeTpl<NewScalar> >&
        casted_integrator_time,
    const std::shared_ptr<DynamicsModelAbstractTpl<NewScalar> >& dynamics) {
  typedef ParameterManagerTpl<NewScalar> ParameterManagerNew;
  typedef ActionModelParamsAbstractTpl<NewScalar> ActionParamsNew;
  typedef IntegratorTimeoptParamsTpl<Scalar> TimeParams;
  typedef IntegratorTimeoptParamsTpl<NewScalar> TimeParamsNew;

  const std::shared_ptr<ParameterManagerNew> dynamics_params =
      internal::getDynamicsParameters(dynamics);
  const std::shared_ptr<StateAbstractTpl<NewScalar> >& state =
      dynamics->get_state();
  const std::shared_ptr<ParameterManagerNew> ret =
      std::make_shared<ParameterManagerNew>(state);
  typename ParameterManagerTpl<Scalar>::ParameterContainer::const_iterator it,
      end;
  for (it = params->get_action_params().begin(),
      end = params->get_action_params().end();
       it != end; ++it) {
    std::shared_ptr<ActionParamsNew> casted_param;
    const std::shared_ptr<TimeParams> time_param =
        std::dynamic_pointer_cast<TimeParams>(it->second->get_param());
    if (time_param != nullptr &&
        time_param->get_integrator_time() == integrator_time) {
      const std::shared_ptr<TimeParamsNew> casted_time_param =
          std::make_shared<TimeParamsNew>(state, casted_integrator_time);
      casted_time_param->set_lb(
          time_param->get_lb().template cast<NewScalar>());
      casted_time_param->set_ub(
          time_param->get_ub().template cast<NewScalar>());
      casted_param = casted_time_param;
    } else {
      casted_param = std::dynamic_pointer_cast<ActionParamsNew>(
          it->second->get_param()->template cast<NewScalar>());
    }
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
    std::shared_ptr<DynamicsParamsAbstractTpl<NewScalar> > casted_param;
    if (dynamics_params != nullptr) {
      typename ParameterManagerNew::ParameterContainer::const_iterator
          casted_it = dynamics_params->get_dynamics_params().find(it->first);
      if (casted_it == dynamics_params->get_dynamics_params().end()) {
        throw_pretty("Invalid call: dynamics parameter '"
                     << it->first << "' is missing after casting");
      }
      casted_param =
          std::dynamic_pointer_cast<DynamicsParamsAbstractTpl<NewScalar> >(
              casted_it->second->get_param());
    } else {
      casted_param =
          std::dynamic_pointer_cast<DynamicsParamsAbstractTpl<NewScalar> >(
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

}  // namespace detail
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_INTEGRATOR_PARAMETER_CAST_HXX_
