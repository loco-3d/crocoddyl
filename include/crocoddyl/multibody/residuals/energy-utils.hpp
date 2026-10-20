///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_ENERGY_UTILS_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_ENERGY_UTILS_HPP_

#include <pinocchio/algorithm/kinematics-derivatives.hpp>

#include "crocoddyl/core/data/observer.hpp"
#include "crocoddyl/core/data/params.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
namespace internal {

template <typename Scalar>
std::shared_ptr<StateMultibodyTpl<Scalar> > getMultibodyState(
    const std::shared_ptr<StateAbstractTpl<Scalar> >& state) {
  std::shared_ptr<StateMultibodyTpl<Scalar> > multibody =
      std::dynamic_pointer_cast<StateMultibodyTpl<Scalar> >(state);
  if (multibody == nullptr) {
    throw_pretty("Invalid argument: state must be StateMultibodyTpl");
  }
  return multibody;
}

template <typename Scalar>
pinocchio::DataTpl<Scalar>* getMultibodyPinocchio(
    DataCollectorAbstractTpl<Scalar>* const shared) {
  DataCollectorMultibodyTpl<Scalar>* multibody =
      dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
  if (multibody == nullptr || multibody->pinocchio == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide DataCollectorMultibodyTpl");
  }
  return multibody->pinocchio;
}

template <typename Scalar>
ParameterDataManagerTpl<Scalar>* getParameterDataManager(
    DataCollectorAbstractTpl<Scalar>* const shared) {
  ParameterDataManagerTpl<Scalar>* manager =
      dynamic_cast<ParameterDataManagerTpl<Scalar>*>(shared);
  if (manager != nullptr) {
    return manager;
  }

  DataCollectorParamsTpl<Scalar>* collector =
      dynamic_cast<DataCollectorParamsTpl<Scalar>*>(shared);
  return collector != nullptr ? collector->parameter_data : nullptr;
}

template <typename Scalar>
MultibodyInertialParamsDataTpl<Scalar>* getInertialParamsData(
    ParameterDataManagerTpl<Scalar>* const manager,
    const std::string& param_name) {
  if (manager == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide ParameterDataManagerTpl");
  }
  typename ParameterDataManagerTpl<Scalar>::ParameterDataContainer::iterator
      it = manager->dynamics_params.find(param_name);
  if (it == manager->dynamics_params.end()) {
    throw_pretty("Invalid argument: dynamics_params does not contain '" +
                 param_name + "'");
  }
  MultibodyInertialParamsDataTpl<Scalar>* inertial =
      dynamic_cast<MultibodyInertialParamsDataTpl<Scalar>*>(it->second.get());
  if (inertial == nullptr) {
    throw_pretty("Invalid argument: '" + param_name +
                 "' is not a MultibodyInertialParamsData");
  }
  return inertial;
}

template <typename Scalar>
std::size_t getDynamicsParamOffset(
    ParameterDataManagerTpl<Scalar>* const manager,
    const std::string& param_name) {
  if (manager == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide ParameterDataManagerTpl");
  }
  return ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(*manager,
                                                                param_name);
}

template <typename Scalar>
void addInertialEnergyRegressor(
    ResidualDataAbstractTpl<Scalar>* const data,
    const MultibodyInertialParamsDataTpl<Scalar>* const inertial,
    const typename pinocchio::DataTpl<Scalar>::RowVectorXs& regressor,
    const std::size_t np_offset, const Scalar scale) {
  for (std::size_t i = 0; i < inertial->dpsi_dp.size(); ++i) {
    data->Rp.block(0, np_offset + 10 * i, 1, 10).noalias() +=
        scale * regressor.template segment<10>(10 * i) * inertial->dpsi_dp[i];
  }
}

template <typename Scalar>
void computeKineticEnergyJacobian(
    const pinocchio::ModelTpl<Scalar>& model, pinocchio::DataTpl<Scalar>& data,
    const Eigen::Ref<const typename MathBaseTpl<Scalar>::VectorXs>& q,
    const Eigen::Ref<const typename MathBaseTpl<Scalar>::VectorXs>& v,
    const Eigen::Ref<const typename MathBaseTpl<Scalar>::VectorXs>& a,
    typename MathBaseTpl<Scalar>::MatrixXs& J,
    typename MathBaseTpl<Scalar>::MatrixXs& Jout,
    typename MathBaseTpl<Scalar>::MatrixXs& dV_dqv,
    typename MathBaseTpl<Scalar>::VectorXs& dT_dqv,
    Eigen::Matrix<Scalar, 6, 1>& tmp6) {
  const std::size_t nv = static_cast<std::size_t>(model.nv);

  pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, a);

  dV_dqv.setZero();
  dT_dqv.setZero();
  for (pinocchio::JointIndex joint_id = 1;
       joint_id < static_cast<pinocchio::JointIndex>(model.njoints);
       ++joint_id) {
    const Eigen::Index row = static_cast<Eigen::Index>(6 * (joint_id - 1));
    J.setZero();
    Jout.setZero();
    pinocchio::getJointVelocityDerivatives(model, data, joint_id,
                                           pinocchio::LOCAL, Jout, J);
    dV_dqv.block(row, 0, 6, static_cast<Eigen::Index>(nv)) = Jout;
    dV_dqv.block(row, static_cast<Eigen::Index>(nv), 6,
                 static_cast<Eigen::Index>(nv)) = J;
  }

  for (pinocchio::JointIndex joint_id = 1;
       joint_id < static_cast<pinocchio::JointIndex>(model.njoints);
       ++joint_id) {
    const Eigen::Index row = static_cast<Eigen::Index>(6 * (joint_id - 1));
    tmp6.noalias() =
        model.inertias[joint_id].matrix() * data.v[joint_id].toVector();
    dT_dqv.noalias() +=
        dV_dqv.block(row, 0, 6, static_cast<Eigen::Index>(2 * nv)).transpose() *
        tmp6;
  }
}

}  // namespace internal
}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_ENERGY_UTILS_HPP_
