///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OBSERVER_BASE_HPP_
#define CROCODDYL_CORE_OBSERVER_BASE_HPP_

#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {
namespace detail {

template <typename Model>
const std::shared_ptr<Model>& check_observer_model(
    const std::shared_ptr<Model>& model) {
  if (model == nullptr) {
    throw_pretty("Invalid argument: model is null");
  }
  return model;
}

template <typename Model>
Model* check_observer_data_model(Model* const model) {
  if (model == nullptr) {
    throw_pretty("Invalid argument: model is null");
  }
  return model;
}

}  // namespace detail

/**
 * @brief Abstract class for observer models
 *
 * An observer model is a specialization of an action model used for optimal
 * estimation problems. It augments the action-model interface with measured
 * torques and an explicit parameter-update entry point.
 *
 * This base layer intentionally stays thin: it adds the observer-specific API
 * while reusing the existing action-model state, cost, constraint, and
 * derivative storage.
 */
template <typename _Scalar>
class ObserverModelAbstractTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ObserverDataAbstractTpl<Scalar> ObserverDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  ObserverModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                           const std::size_t ntau, const std::size_t nu,
                           const std::size_t nr = 0, const std::size_t ng = 0,
                           const std::size_t nh = 0, const std::size_t ng_T = 0,
                           const std::size_t nh_T = 0,
                           const std::size_t np = 0);
  virtual ~ObserverModelAbstractTpl() = default;

  /**
   * @brief Update the value of the observer parameters
   *
   * This method is observer-specific and intentionally remains separate from
   * the parameter-manager infrastructure added for action and dynamics models.
   *
   * @param[in] data  Observer action data
   * @param[in] p     Observer parameter vector
   */
  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override = 0;

  /**
   * @brief Wire a parameter manager into this observer model and its data
   */
  virtual void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                          std::shared_ptr<ParameterManager> params) override;

  /**
   * @brief Update the observed torques
   *
   * @param[in] tau_meas  Measured torque vector
   */
  virtual void update_tau(const Eigen::Ref<const VectorXs>& tau_meas);

  /**
   * @brief Create observer action data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;

  /**
   * @brief Create observer action data with optional external parameter data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /**
   * @brief Checks that a specific action data belongs to this observer model
   */
  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  /**
   * @brief Return the dimension of the measured torque vector
   */
  std::size_t get_ntau() const;

  /**
   * @brief Return the measured torques
   */
  const VectorXs& get_tau_meas() const;

 protected:
  using Base::np_;

  std::size_t ntau_;   //!< Measured torque dimension
  VectorXs tau_meas_;  //!< Measured torques
};

template <typename _Scalar>
struct ObserverDataAbstractTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ObserverDataAbstractTpl(Model<Scalar>* const model)
      : Base(detail::check_observer_data_model(model)),
        dissipative_E(1),
        dE_dv(1, model->get_state()->get_nv()),
        dE_dp(1, model->get_np()) {
    setZero();
  }
  virtual ~ObserverDataAbstractTpl() = default;

  template <class Model>
  void resize(Model* const model, const bool running_node = true) {
    Base::resize(model, running_node);
    dissipative_E.resize(1);
    dE_dv.resize(1, model->get_state()->get_nv());
    dE_dp.resize(1, model->get_np());
    setZero();
  }

  virtual void setZero() override {
    Base::setZero();
    dissipative_E.setZero();
    dE_dv.setZero();
    dE_dp.setZero();
  }

  VectorXs dissipative_E;  //!< Integrated dissipative energy
  MatrixXs dE_dv;          //!< Jacobian of dissipative energy w.r.t. velocity
  MatrixXs dE_dp;          //!< Jacobian of dissipative energy w.r.t. parameters
};

}  // namespace crocoddyl

#include "crocoddyl/core/observer-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ObserverModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ObserverDataAbstractTpl)

#endif  // CROCODDYL_CORE_OBSERVER_BASE_HPP_
