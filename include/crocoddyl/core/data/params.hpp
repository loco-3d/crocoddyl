///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_PARAMS_HPP_
#define CROCODDYL_CORE_DATA_PARAMS_HPP_

#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

/**
 * @brief Shared parameter data payload
 *
 * This caller-owned payload stores the active parameter vector as a contiguous
 * action-parameter prefix followed by a dynamics-parameter suffix. It owns the
 * parameter vector and the corresponding action and torque sensitivity
 * buffers. Resizing updates both partition dimensions, preserves the active
 * status and zeros all buffers.
 */
template <typename _Scalar>
struct ParamsDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the shared parameter payload
   *
   * @param[in] state        State description used to size the sensitivities
   * @param[in] np_action    Action-parameter prefix dimension
   * @param[in] np_dynamics  Dynamics-parameter suffix dimension
   */
  ParamsDataAbstractTpl(std::shared_ptr<StateAbstract> state,
                        const std::size_t np_action = 0,
                        const std::size_t np_dynamics = 0)
      : np(np_action + np_dynamics),
        np_action(np_action),
        np_dynamics(np_dynamics),
        p(np_action + np_dynamics),
        dx_dp(state->get_ndx(), np_action),
        dtau_dp(state->get_nv(), np_dynamics),
        active(true),
        ndx_(state->get_ndx()),
        nv_(state->get_nv()) {
    setZero();
  }
  virtual ~ParamsDataAbstractTpl() {}

  /**
   * @brief Resize and zero the action/dynamics parameter partitions
   *
   * The active status is intentionally left unchanged.
   */
  virtual void resize(const std::size_t np_action,
                      const std::size_t np_dynamics) {
    this->np_action = np_action;
    this->np_dynamics = np_dynamics;
    np = np_action + np_dynamics;
    p.resize(np);
    dx_dp.resize(ndx_, np_action);
    dtau_dp.resize(nv_, np_dynamics);
    setZero();
  }

  /**
   * @brief Zero the parameter and sensitivity buffers
   *
   * The partition dimensions and active status are left unchanged.
   */
  virtual void setZero() {
    p.setZero();
    dx_dp.setZero();
    dtau_dp.setZero();
  }

  std::size_t np;           //!< Total parameter dimension
  std::size_t np_action;    //!< Action-parameter dimension
  std::size_t np_dynamics;  //!< Dynamics-parameter dimension
  VectorXs p;      //!< Active parameter vector: action prefix, dynamics suffix
  MatrixXs dx_dp;  //!< Action sensitivity Jacobian
  MatrixXs dtau_dp;  //!< Dynamics torque-regressor Jacobian
  bool active;       //!< Activation status

 protected:
  std::size_t ndx_;
  std::size_t nv_;
};

/**
 * @brief Action-parameter data payload
 *
 * This caller-owned specialization places its complete parameter vector in
 * the action prefix and leaves the dynamics suffix empty.
 */
template <typename _Scalar>
struct ActionModelParamsDataAbstractTpl
    : public ParamsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ParamsDataAbstractTpl<Scalar> Base;
  typedef StateAbstractTpl<Scalar> StateAbstract;

  /**
   * @brief Initialize the action-parameter payload
   *
   * @param[in] state  State description used to size the sensitivity
   * @param[in] np     Action-parameter dimension
   */
  ActionModelParamsDataAbstractTpl(std::shared_ptr<StateAbstract> state,
                                   const std::size_t np = 0)
      : Base(state, np, 0) {}
  virtual ~ActionModelParamsDataAbstractTpl() {}
};

/**
 * @brief Dynamics-parameter data payload
 *
 * This caller-owned specialization places its complete parameter vector in
 * the dynamics suffix and leaves the action prefix empty. It owns the
 * parameter vector and the \f$nv\times np\f$ joint-torque regressor buffer.
 * Resizing and zeroing follow the inherited payload semantics and preserve
 * the active status. It stores no model pointer; the caller controls its
 * lifetime.
 */
template <typename _Scalar>
struct DynamicsParamsDataAbstractTpl : public ParamsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ParamsDataAbstractTpl<Scalar> Base;
  typedef StateAbstractTpl<Scalar> StateAbstract;

  /**
   * @brief Initialize the dynamics-parameter payload
   *
   * @param[in] state  Shared state description used to size \f$dtau/dp\f$
   * @param[in] np     Dynamics-parameter dimension
   */
  DynamicsParamsDataAbstractTpl(std::shared_ptr<StateAbstract> state,
                                const std::size_t np = 0)
      : Base(state, 0, np) {}
  virtual ~DynamicsParamsDataAbstractTpl() {}
};

/**
 * @brief Collector for shared parameter data
 *
 * The collector shares ownership of the parameter payload. Its optional
 * ParameterDataManagerTpl link is non-owning and, when non-null, the manager
 * must outlive every collector that references it.
 */
template <typename Scalar>
struct DataCollectorParamsTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the parameter-data collector
   *
   * @param[in] params          Shared parameter payload
   * @param[in] parameter_data  Optional non-owning parameter-manager data link
   */
  DataCollectorParamsTpl(
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorAbstractTpl<Scalar>(),
        params(params),
        parameter_data(parameter_data) {}
  virtual ~DataCollectorParamsTpl() {}

  std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params;
  ParameterDataManagerTpl<Scalar>* parameter_data;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorParamsTpl)

#endif  // CROCODDYL_CORE_DATA_PARAMS_HPP_
