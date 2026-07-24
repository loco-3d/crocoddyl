///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_PARAMS_ACTUATION_HPP_
#define CROCODDYL_MULTIBODY_PARAMS_ACTUATION_HPP_

#include "crocoddyl/core/params-base.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

/**
 * @brief Dynamics-parameter model for multibody actuation
 *
 * The dynamics partition contains the concatenated joint parameter vector.
 * update() synchronizes each joint model and its internal parameterization;
 * computeJointTorqueRegressor() fills \f$d\tau/dp\in
 * \mathbb{R}^{n_v\times n_p}\f$. The wrapped actuation model is shared.
 */
template <typename _Scalar>
class ActuationMultibodyParamsTpl : public DynamicsParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsParamsAbstractTpl<Scalar> Base;
  typedef ParamsDataAbstractTpl<Scalar> ParamsDataAbstract;
  typedef ActuationModelMultibodyTpl<Scalar> ActuationModelMultibody;
  typedef ActuationMultibodyParamsDataTpl<Scalar> ActuationMultibodyParamsData;
  typedef typename Base::DynamicsDataAbstract DynamicsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  CROCODDYL_DERIVED_CAST(ParamsModelBase, ActuationMultibodyParamsTpl)

  /**
   * @brief Initialize dynamics parameters for a multibody actuation model
   *
   * The actuation model is retained through shared ownership. Its joint-model
   * parameter order defines this model's dynamics partition and bounds.
   *
   * @param[in] actuation Shared multibody actuation model
   */
  explicit ActuationMultibodyParamsTpl(
      std::shared_ptr<ActuationModelMultibody> actuation);
  virtual ~ActuationMultibodyParamsTpl() = default;

  /**
   * @brief Update all joint models from an unconstrained parameter vector
   *
   * The data stores \f$p\f$, physical values \f$\gamma\f$ and
   * \f$d\gamma/dp\f$ in the actuation model's parameter order.
   *
   * @param[in,out] data Specialized parameter data
   * @param[in] p Unconstrained dynamics parameters
   */
  virtual void update(const std::shared_ptr<ParamsDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) override;

  /**
   * @brief Fill inherited \f$d\tau/dp\f$ from the joint-model regressors
   *
   * @param[in,out] data Dynamics data associated with the evaluation
   * @param[in,out] params Specialized parameter data receiving the regressor
   * @param[in] x Multibody state
   * @param[in] u Concatenated joint commands
   */
  virtual void computeJointTorqueRegressor(
      const std::shared_ptr<DynamicsDataAbstract>& data,
      const std::shared_ptr<ParamsDataAbstract>& params,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Create specialized data with an empty action partition */
  virtual std::shared_ptr<ParamsDataAbstract> createData() override;

  /** @brief Check specialized runtime type and inherited dimensions */
  virtual bool checkData(
      const std::shared_ptr<ParamsDataAbstract>& data) const override;

  /** @brief Return the retained shared multibody actuation model */
  const std::shared_ptr<ActuationModelMultibody>& get_actuation() const;

  /** @brief Cast the parameter and actuation models to another scalar */
  template <typename NewScalar>
  ActuationMultibodyParamsTpl<NewScalar> cast() const;

  /** @brief Print the multibody-actuation parameter model */
  virtual void print(std::ostream& os) const override;

 protected:
  std::shared_ptr<ActuationMultibodyParamsData> castData(
      const std::shared_ptr<ParamsDataAbstract>& data) const;

  std::shared_ptr<ActuationModelMultibody> actuation_;
};

/**
 * @brief Parameter data for multibody actuation
 *
 * This specialized dynamics payload has \f$n_{p,\mathrm{action}}=0\f$ and
 * \f$n_{p,\mathrm{dynamics}}=n_p\f$. It inherits \f$p\f$, active state,
 * offsets and \f$d\tau/dp\f$, and stores internal parameters \f$\gamma\f$ and
 * \f$d\gamma/dp\f$. resize() preserves this partition and setZero() clears all
 * inherited and specialized derivative storage.
 */
template <typename _Scalar>
struct ActuationMultibodyParamsDataTpl
    : public DynamicsParamsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsParamsDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Allocate specialized actuation-parameter data
   *
   * The model supplies dimensions only and is not retained. The resulting
   * payload owns its action/dynamics partition and derivative buffers.
   *
   * @param[in] model Multibody-actuation parameter model
   */
  template <template <typename Scalar> class Model>
  explicit ActuationMultibodyParamsDataTpl(Model<Scalar>* const model)
      : Base(model->get_state(), model->get_np()),
        gamma(model->get_np()),
        dgamma_dp(model->get_np(), model->get_np()) {
    setZero();
  }
  virtual ~ActuationMultibodyParamsDataTpl() = default;

  /**
   * @brief Resize the action/dynamics partition and specialized buffers
   *
   * This family uses \f$n_{p,\mathrm{action}}=0\f$ and
   * \f$n_{p,\mathrm{dynamics}}=n_p\f$; dimensions are validated by the model.
   *
   * @param[in] np_action Action-parameter dimension
   * @param[in] np_dynamics Dynamics-parameter dimension
   */
  virtual void resize(const std::size_t np_action,
                      const std::size_t np_dynamics) override {
    Base::resize(np_action, np_dynamics);
    gamma.resize(this->np_dynamics);
    dgamma_dp.resize(this->np_dynamics, this->np_dynamics);
    setZero();
  }

  /** @brief Zero inherited storage, physical parameters and their Jacobian */
  virtual void setZero() override {
    Base::setZero();
    gamma.setZero();
    dgamma_dp.setZero();
  }

  VectorXs gamma;      //!< Physical joint-model parameters
  MatrixXs dgamma_dp;  //!< Physical-parameter Jacobian
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/params/actuation.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActuationMultibodyParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActuationMultibodyParamsDataTpl)

#endif  // CROCODDYL_MULTIBODY_PARAMS_ACTUATION_HPP_
