///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_POWER_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_POWER_HPP_

#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/regressor.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <string>

#include "crocoddyl/core/data/observer.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/residuals/energy-utils.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief One-step mechanical-energy balance residual
 *
 * The running residual is \f$r=E(x^+)-E(x)+E_d-P^*\f$, using an observer
 * payload prepared by Euler integration. Missing running observation values or
 * derivatives produce zero outputs. Optional active inertial and actuation
 * parameter entries contribute direct D075 parameter derivatives. The
 * terminal residual and derivatives are zero.
 */
template <typename _Scalar>
class ResidualModelPowerTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelPowerTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataPowerTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef DataCollectorObserverTpl<Scalar> DataCollectorObserver;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef ActuationMultibodyParamsDataTpl<Scalar> ActuationMultibodyParamsData;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6s;

  /** @brief Construct the power residual and optionally name parameter data. */
  ResidualModelPowerTpl(std::shared_ptr<typename Base::StateAbstract> state,
                        const std::size_t nu, const std::size_t np,
                        const Scalar P_ref = Scalar(0),
                        const std::string& inertial_param_name = "",
                        const std::string& actuation_param_name = "");
  virtual ~ResidualModelPowerTpl() = default;

  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  template <typename NewScalar>
  ResidualModelPowerTpl<NewScalar> cast() const;

  /** @brief Return the reference energy increment. */
  Scalar get_reference() const;
  /** @brief Set the reference energy increment. */
  void set_reference(const Scalar reference);
  /** @brief Return the selected inertial parameter name. */
  const std::string& get_inertial_param_name() const;
  /** @brief Return the selected actuation parameter name. */
  const std::string& get_actuation_param_name() const;
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::np_;
  using Base::nr_;
  using Base::nu_;
  using Base::state_;

 private:
  Scalar P_ref_;
  std::string inertial_param_name_;
  std::string actuation_param_name_;
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Data for ResidualModelPowerTpl
 *
 * Collector pointers are non-owning and must remain valid. Numerical
 * workspaces are owned by the data and reused without allocation.
 */
template <typename _Scalar>
struct ResidualDataPowerTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef DataCollectorObserverTpl<Scalar> DataCollectorObserver;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef ActuationMultibodyParamsDataTpl<Scalar> ActuationMultibodyParamsData;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6s;

  template <template <typename S> class Model>
  ResidualDataPowerTpl(Model<Scalar>* const model,
                       DataCollectorAbstract* const data)
      : Base(model, data),
        pinocchio(internal::getMultibodyPinocchio<Scalar>(data)),
        observer(dynamic_cast<DataCollectorObserver*>(data)),
        parameter_data(internal::getParameterDataManager<Scalar>(data)),
        inertial_data(nullptr),
        actuation_data(nullptr),
        inertial_np_offset(0),
        J(6, model->get_state()->get_nv()),
        Jout(6, model->get_state()->get_nv()),
        dV_dqv(6 * model->get_state()->get_nv(), model->get_state()->get_ndx()),
        dK_dqv(model->get_state()->get_ndx()),
        dTm_dqv(model->get_state()->get_ndx()),
        dTp_dqv(model->get_state()->get_ndx()),
        dEp_dx(model->get_state()->get_ndx()),
        dUm_dq(model->get_state()->get_nv()),
        dUp_dq(model->get_state()->get_nv()),
        tmp6(Vector6s::Zero()) {
    if (parameter_data != nullptr) {
      if (parameter_data->params == nullptr) {
        throw_pretty("Invalid argument: aggregate parameter data is null");
      }
      if (static_cast<std::size_t>(parameter_data->params->np) !=
          model->get_np()) {
        throw_pretty(
            "Invalid argument: ParameterDataManager np does not "
            "match the residual model");
      }
    }
    if (!model->get_inertial_param_name().empty()) {
      if (parameter_data == nullptr) {
        throw_pretty(
            "Invalid argument: named inertial parameters require "
            "ParameterDataManagerTpl");
      }
      inertial_data = internal::getInertialParamsData<Scalar>(
          parameter_data, model->get_inertial_param_name());
      inertial_np_offset = internal::getDynamicsParamOffset<Scalar>(
          parameter_data, model->get_inertial_param_name());
    }
    if (!model->get_actuation_param_name().empty()) {
      if (parameter_data == nullptr) {
        throw_pretty(
            "Invalid argument: named actuation parameters require "
            "ParameterDataManagerTpl");
      }
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *parameter_data, model->get_actuation_param_name());
      typename ParameterDataManager::ParameterDataContainer::iterator it =
          parameter_data->dynamics_params.find(
              model->get_actuation_param_name());
      if (it == parameter_data->dynamics_params.end()) {
        throw_pretty("Invalid argument: dynamics_params does not contain '" +
                     model->get_actuation_param_name() + "'");
      }
      actuation_data =
          dynamic_cast<ActuationMultibodyParamsData*>(it->second.get());
      if (actuation_data == nullptr) {
        throw_pretty("Invalid argument: '" + model->get_actuation_param_name() +
                     "' is not an ActuationMultibodyParamsData");
      }
    }
    J.setZero();
    Jout.setZero();
    dV_dqv.setZero();
    dK_dqv.setZero();
    dTm_dqv.setZero();
    dTp_dqv.setZero();
    dEp_dx.setZero();
    dUm_dq.setZero();
    dUp_dq.setZero();
  }
  virtual ~ResidualDataPowerTpl() = default;

  using Base::r;
  using Base::Rp;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  pinocchio::DataTpl<Scalar>* pinocchio;
  DataCollectorObserver* observer;
  ParameterDataManager* parameter_data;
  MultibodyInertialParamsData* inertial_data;
  ActuationMultibodyParamsData* actuation_data;
  std::size_t inertial_np_offset;
  MatrixXs J;
  MatrixXs Jout;
  MatrixXs dV_dqv;
  VectorXs dK_dqv;
  VectorXs dTm_dqv;
  VectorXs dTp_dqv;
  VectorXs dEp_dx;
  VectorXs dUm_dq;
  VectorXs dUp_dq;
  Vector6s tmp6;
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/residuals/power.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ResidualModelPowerTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ResidualDataPowerTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_POWER_HPP_
