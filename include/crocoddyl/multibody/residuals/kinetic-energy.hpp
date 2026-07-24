///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_KINETIC_ENERGY_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_KINETIC_ENERGY_HPP_

#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/regressor.hpp>
#include <string>

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/residuals/energy-utils.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Kinetic-energy residual \f$r=T(q,v,p)-T^*\f$
 *
 * Rx is the exact state derivative. When \c param_name names an active
 * MultibodyInertialParams entry, Rp is assembled at its D075 offset. Running
 * and terminal evaluations are identical.
 */
template <typename _Scalar>
class ResidualModelKineticEnergyTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelKineticEnergyTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataKineticEnergyTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6s;

  /** @brief Construct the residual and optionally select inertial parameters.
   */
  ResidualModelKineticEnergyTpl(
      std::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu,
      const std::size_t np, const Scalar T_ref = Scalar(0),
      const std::string& param_name = "");
  virtual ~ResidualModelKineticEnergyTpl() = default;

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
  ResidualModelKineticEnergyTpl<NewScalar> cast() const;

  /** @brief Return the kinetic-energy reference. */
  Scalar get_reference() const;
  /** @brief Set the kinetic-energy reference. */
  void set_reference(const Scalar reference);
  /** @brief Return the selected inertial-parameter name, or an empty string. */
  const std::string& get_param_name() const;
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::np_;
  using Base::nr_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  Scalar T_ref_;
  std::string param_name_;
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Data for ResidualModelKineticEnergyTpl
 *
 * Shared collector pointers are non-owning. Jacobian workspaces are owned by
 * each data object and reused without allocation.
 */
template <typename _Scalar>
struct ResidualDataKineticEnergyTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6s;

  template <template <typename S> class Model>
  ResidualDataKineticEnergyTpl(Model<Scalar>* const model,
                               DataCollectorAbstract* const data)
      : Base(model, data),
        pinocchio(internal::getMultibodyPinocchio<Scalar>(data)),
        parameter_data(internal::getParameterDataManager<Scalar>(data)),
        inertial_data(nullptr),
        np_offset(0),
        J(6, model->get_state()->get_nv()),
        Jout(6, model->get_state()->get_nv()),
        dV_dqv(6 * model->get_state()->get_nv(), model->get_state()->get_ndx()),
        dT_dqv(model->get_state()->get_ndx()),
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
    if (!model->get_param_name().empty()) {
      if (parameter_data == nullptr) {
        throw_pretty(
            "Invalid argument: named inertial parameters require "
            "ParameterDataManagerTpl");
      }
      inertial_data = internal::getInertialParamsData<Scalar>(
          parameter_data, model->get_param_name());
      np_offset = internal::getDynamicsParamOffset<Scalar>(
          parameter_data, model->get_param_name());
    }
    J.setZero();
    Jout.setZero();
    dV_dqv.setZero();
    dT_dqv.setZero();
  }
  virtual ~ResidualDataKineticEnergyTpl() = default;

  using Base::r;
  using Base::Rp;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  pinocchio::DataTpl<Scalar>* pinocchio;
  ParameterDataManager* parameter_data;
  MultibodyInertialParamsData* inertial_data;
  std::size_t np_offset;
  MatrixXs J;
  MatrixXs Jout;
  MatrixXs dV_dqv;
  VectorXs dT_dqv;
  Vector6s tmp6;
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/residuals/kinetic-energy.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelKineticEnergyTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataKineticEnergyTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_KINETIC_ENERGY_HPP_
