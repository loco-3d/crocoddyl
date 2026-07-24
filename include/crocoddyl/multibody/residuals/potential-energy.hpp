///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_POTENTIAL_ENERGY_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_POTENTIAL_ENERGY_HPP_

#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/regressor.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <string>

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/residuals/energy-utils.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Potential-energy residual \f$r=U(q,p)-U^*\f$
 *
 * State derivatives follow generalized gravity. When \c param_name names an
 * active MultibodyInertialParams entry, Rp is assembled at its D075 offset.
 * Running and terminal evaluations are identical.
 */
template <typename _Scalar>
class ResidualModelPotentialEnergyTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelPotentialEnergyTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataPotentialEnergyTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef typename MathBase::VectorXs VectorXs;

  /** @brief Construct the residual and optionally select inertial parameters.
   */
  ResidualModelPotentialEnergyTpl(
      std::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu,
      const std::size_t np, const Scalar U_ref = Scalar(0),
      const std::string& param_name = "");
  virtual ~ResidualModelPotentialEnergyTpl() = default;

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
  ResidualModelPotentialEnergyTpl<NewScalar> cast() const;

  /** @brief Return the potential-energy reference. */
  Scalar get_reference() const;
  /** @brief Set the potential-energy reference. */
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
  Scalar U_ref_;
  std::string param_name_;
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Data for ResidualModelPotentialEnergyTpl
 *
 * Pinocchio and parameter pointers are non-owning views of the shared
 * collector, which must outlive this data.
 */
template <typename _Scalar>
struct ResidualDataPotentialEnergyTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;

  template <template <typename S> class Model>
  ResidualDataPotentialEnergyTpl(Model<Scalar>* const model,
                                 DataCollectorAbstract* const data)
      : Base(model, data),
        pinocchio(internal::getMultibodyPinocchio<Scalar>(data)),
        parameter_data(internal::getParameterDataManager<Scalar>(data)),
        inertial_data(nullptr),
        np_offset(0) {
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
  }
  virtual ~ResidualDataPotentialEnergyTpl() = default;

  using Base::r;
  using Base::Rp;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  pinocchio::DataTpl<Scalar>* pinocchio;
  ParameterDataManager* parameter_data;
  MultibodyInertialParamsData* inertial_data;
  std::size_t np_offset;
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/residuals/potential-energy.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelPotentialEnergyTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataPotentialEnergyTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_POTENTIAL_ENERGY_HPP_
