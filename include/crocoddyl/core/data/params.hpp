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

namespace crocoddyl {

/**
 * @brief Shared parameter data payload
 *
 * This caller-owned payload stores the active parameter vector as a contiguous
 * action-parameter prefix followed by a dynamics-parameter suffix. It owns the
 * parameter vector and parameter-dependent quantities computed by derived
 * classes. Derivative workspaces deliberately live in the action or dynamics
 * data associated with each shooting node, so this payload can be shared
 * safely by all nodes in a parameter phase.
 */
template <typename _Scalar>
struct ParamsDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the shared parameter payload
   *
   * @param[in] np_action    Action-parameter prefix dimension
   * @param[in] np_dynamics  Dynamics-parameter suffix dimension
   */
  explicit ParamsDataAbstractTpl(const std::size_t np_action = 0,
                                 const std::size_t np_dynamics = 0)
      : np(np_action + np_dynamics),
        np_action(np_action),
        np_dynamics(np_dynamics),
        p(np_action + np_dynamics),
        active(true) {
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
    setZero();
  }

  /**
   * @brief Zero the parameter vector
   *
   * The partition dimensions and active status are left unchanged.
   */
  virtual void setZero() { p.setZero(); }

  std::size_t np;           //!< Total parameter dimension
  std::size_t np_action;    //!< Action-parameter dimension
  std::size_t np_dynamics;  //!< Dynamics-parameter dimension
  VectorXs p;   //!< Active parameter vector: action prefix, dynamics suffix
  bool active;  //!< Activation status
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

  /**
   * @brief Initialize the action-parameter payload
   *
   * @param[in] np     Action-parameter dimension
   */
  explicit ActionModelParamsDataAbstractTpl(const std::size_t np = 0)
      : Base(np, 0) {}
  virtual ~ActionModelParamsDataAbstractTpl() {}
};

/**
 * @brief Dynamics-parameter data payload
 *
 * This caller-owned specialization places its complete parameter vector in
 * the dynamics suffix and leaves the action prefix empty. It owns the
 * parameter vector. Resizing and zeroing follow the inherited payload
 * semantics and preserve the active status. It stores no model pointer; the
 * caller controls its lifetime.
 */
template <typename _Scalar>
struct DynamicsParamsDataAbstractTpl : public ParamsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ParamsDataAbstractTpl<Scalar> Base;

  /**
   * @brief Initialize the dynamics-parameter payload
   *
   * @param[in] np     Dynamics-parameter dimension
   */
  explicit DynamicsParamsDataAbstractTpl(const std::size_t np = 0)
      : Base(0, np) {}
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
