///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_PARAMS_BASE_HPP_
#define CROCODDYL_CORE_PARAMS_BASE_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/data/params.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"

namespace crocoddyl {

class ParamsModelBase {
 public:
  virtual ~ParamsModelBase() = default;

  CROCODDYL_BASE_CAST(ParamsModelBase, ParamsAbstractTpl)
};

/**
 * @brief Abstract interface for parameter models
 *
 * A parameter model owns its state description, dimension and bounds. It
 * samples a local parameter vector and updates a caller-owned shared parameter
 * payload. The default update is a no-op; derived models can override it.
 */
template <typename _Scalar>
class ParamsAbstractTpl : public ParamsModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ParamsDataAbstractTpl<Scalar> ParamsDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the parameter model
   *
   * @param[in] state  Shared state description
   * @param[in] np     Dimension of the parameter vector
   */
  ParamsAbstractTpl(std::shared_ptr<StateAbstract> state,
                    const std::size_t np = 0);
  virtual ~ParamsAbstractTpl() = default;

  /**
   * @brief Update the parameter data with a parameter vector
   *
   * @param[in] data  Parameter data owned by the caller
   * @param[in] p     Parameter vector
   */
  virtual void update(const std::shared_ptr<ParamsDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p);

  /**
   * @brief Create an action-partitioned parameter payload
   */
  virtual std::shared_ptr<ParamsDataAbstract> createData();

  /**
   * @brief Check if a given data has this model's parameter dimension
   */
  virtual bool checkData(const std::shared_ptr<ParamsDataAbstract>& data) const;

  /**
   * @brief Return a zero parameter vector
   */
  virtual VectorXs zero() const;

  /**
   * @brief Return a random parameter vector in [0, 1]
   */
  virtual VectorXs rand() const;

  /**
   * @brief Return the state description
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the dimension of the parameter vector
   */
  std::size_t get_np() const;

  /**
   * @brief Return the lower bound of the parameter vector
   */
  const VectorXs& get_lb() const;

  /**
   * @brief Return the upper bound of the parameter vector
   */
  const VectorXs& get_ub() const;

  /**
   * @brief Set the lower bound of the parameter vector
   *
   * @param[in] lb  Lower bound vector
   */
  void set_lb(const VectorXs& lb);

  /**
   * @brief Set the upper bound of the parameter vector
   *
   * @param[in] ub  Upper bound vector
   */
  void set_ub(const VectorXs& ub);

  /**
   * @brief Print information about the parameter model
   */
  virtual void print(std::ostream& os) const;

  virtual std::shared_ptr<ParamsModelBase> cloneAsDouble() const override;
  virtual std::shared_ptr<ParamsModelBase> cloneAsFloat() const override;
#ifdef CROCODDYL_WITH_CODEGEN
  virtual std::shared_ptr<ParamsModelBase> cloneAsADDouble() const override;
  virtual std::shared_ptr<ParamsModelBase> cloneAsADFloat() const override;
#endif

  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const ParamsAbstractTpl<Scalar>& model);

 protected:
  std::size_t np_;
  std::shared_ptr<StateAbstract> state_;
  VectorXs lb_;
  VectorXs ub_;
};

/**
 * @brief Abstract action-parameter model
 *
 * This interface extends ParamsAbstractTpl with the computation of the state
 * sensitivity to its action-parameter vector. It creates caller-owned data
 * whose parameter vector occupies the action prefix of the shared payload.
 */
template <typename _Scalar>
class ActionModelParamsAbstractTpl : public ParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ParamsAbstractTpl<Scalar> Base;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  /**
   * @brief Initialize the action-parameter model
   *
   * @param[in] state  Shared state description
   * @param[in] np     Dimension of the action-parameter vector
   */
  ActionModelParamsAbstractTpl(std::shared_ptr<StateAbstract> state,
                               const std::size_t np = 0);
  virtual ~ActionModelParamsAbstractTpl() = default;

  /**
   * @brief Compute action sensitivity with respect to parameters
   *
   * @param[in] data    Action data owned by the caller
   * @param[in] params  Shared parameter payload
   * @param[in] x       State point
   * @param[in] u       Control point
   */
  virtual void computeParamSensitivity(
      const std::shared_ptr<ActionDataAbstract>& data,
      const std::shared_ptr<ParamsDataAbstract>& params,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Create the action-partitioned parameter data
   */
  virtual std::shared_ptr<ParamsDataAbstract> createData() override;
};

/**
 * @brief Abstract dynamics-parameter model
 *
 * This interface extends ParamsAbstractTpl with the computation of the joint
 * torque sensitivity to its dynamics-parameter vector. It creates
 * caller-owned data whose parameter vector occupies the dynamics suffix of
 * the shared payload.
 */
template <typename _Scalar>
class DynamicsParamsAbstractTpl : public ParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ParamsAbstractTpl<Scalar> Base;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  /**
   * @brief Initialize the dynamics-parameter model
   *
   * @param[in] state  Shared state description
   * @param[in] np     Dimension of the dynamics-parameter vector
   */
  DynamicsParamsAbstractTpl(std::shared_ptr<StateAbstract> state,
                            const std::size_t np = 0);
  virtual ~DynamicsParamsAbstractTpl() = default;

  /**
   * @brief Compute the joint-torque regressor with respect to parameters
   *
   * @param[in] data    Dynamics data owned by the caller
   * @param[in] params  Shared dynamics-parameter payload
   * @param[in] x       State point
   * @param[in] u       Control point
   */
  virtual void computeJointTorqueRegressor(
      const std::shared_ptr<DynamicsDataAbstract>& data,
      const std::shared_ptr<ParamsDataAbstract>& params,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Create the dynamics-partitioned parameter data
   *
   * The return type remains the generic parameter-data pointer while the
   * allocated object is DynamicsParamsDataAbstractTpl.
   */
  virtual std::shared_ptr<ParamsDataAbstract> createData() override;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/params-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ParamsAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActionModelParamsAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::DynamicsParamsAbstractTpl)

#endif  // CROCODDYL_CORE_PARAMS_BASE_HPP_
