///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_MULTIBODY_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_MULTIBODY_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/multibody/actuations/joint-dynamics-base.hpp"
#include "crocoddyl/multibody/actuations/joint-friction.hpp"
#include "crocoddyl/multibody/actuations/joint-identity.hpp"
#include "crocoddyl/multibody/actuations/joint-thruster.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Runtime data for a multibody stack of joint-dynamics models
 *
 * In addition to the standard actuation data, this stores accumulated
 * generalized friction and one independently allocated data object for every
 * joint model, indexed by Pinocchio joint id and insertion order.
 */
template <typename _Scalar>
struct ActuationDataMultibodyTpl : public ActuationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationDataAbstractTpl<Scalar> Base;
  typedef JointDynamicsDataAbstractTpl<Scalar> JointDynamicsDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Allocate multibody actuation data
   *
   * The model supplies dimensions only; createData() subsequently allocates
   * and owns one joint payload per shared joint model.
   *
   * @param[in] model Multibody actuation model
   */
  template <template <typename Scalar> class Model>
  explicit ActuationDataMultibodyTpl(Model<Scalar>* const model)
      : Base(model), friction(model->get_state()->get_nv()) {
    friction.setZero();
  }
  virtual ~ActuationDataMultibodyTpl() = default;

  using Base::dtau_du;
  using Base::dtau_dx;
  using Base::Mtau;
  using Base::tau;
  using Base::tau_set;
  using Base::u;

  VectorXs friction;  //!< Accumulated generalized friction torque
  std::vector<std::vector<std::shared_ptr<JointDynamicsDataAbstract> > >
      joint;  //!< Owned joint data, indexed by joint id and insertion order
};

/**
 * @brief Assemble multibody actuation from per-joint dynamics models
 *
 * Commands are concatenated in Pinocchio joint order and, within a joint, in
 * insertion order. Parameter blocks follow the same ordering. The default
 * constructor reproduces floating-base actuation with identity models on all
 * actuated joints. Each data object owns its joint runtime payloads, while the
 * model shares the supplied joint models.
 */
template <typename _Scalar>
class ActuationModelMultibodyTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActuationModelBase, ActuationModelMultibodyTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef ActuationDataMultibodyTpl<Scalar> ActuationDataMultibody;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef JointDynamicsModelAbstractTpl<Scalar> JointDynamicsModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the legacy-compatible default multibody actuation
   *
   * Identity dynamics are created once for every actuated Pinocchio joint.
   *
   * @param[in] state Multibody state shared by the actuation model
   */
  explicit ActuationModelMultibodyTpl(std::shared_ptr<StateMultibody> state);

  /**
   * @brief Initialize multibody actuation from per-joint dynamics models
   *
   * Models are grouped by Pinocchio joint id and insertion order. Models on a
   * joint share one command block and must have compatible dimensions.
   *
   * @param[in] state Multibody state shared by the actuation model
   * @param[in] joints Joint models retained through shared ownership
   */
  ActuationModelMultibodyTpl(
      std::shared_ptr<StateMultibody> state,
      const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joints);
  virtual ~ActuationModelMultibodyTpl() = default;

  /** @brief Compute and accumulate joint torques and friction */
  virtual void calc(const std::shared_ptr<ActuationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute and assemble joint-torque derivatives
   *
   * calc() must be called first so every joint payload contains its cached
   * lifecycle quantities.
   */
  virtual void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Compute the concatenated commands for a desired torque vector */
  virtual void commands(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& tau) override;

  /** @brief Assemble the command transform from the joint inverse maps */
  virtual void torqueTransform(
      const std::shared_ptr<ActuationDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Create independent multibody and per-joint runtime data */
  virtual std::shared_ptr<ActuationDataAbstract> createData() override;

  /**
   * @brief Return shared joint models grouped by joint id and insertion order
   */
  const std::vector<std::vector<std::shared_ptr<JointDynamicsModelAbstract> > >&
  get_joints() const;

  /** @brief Return the total number of joint-dynamics parameters */
  std::size_t get_np() const;

  /**
   * @brief Update joint parameters and collect their physical values
   * @param[in] p Concatenated unconstrained parameters in joint-model order
   * @param[out] gamma Concatenated physical parameters in the same order
   */
  void update_p(const Eigen::Ref<const VectorXs>& p,
                Eigen::Ref<VectorXs> gamma);

  /**
   * @brief Assemble the block-diagonal physical-parameter Jacobian
   * @param[out] dgamma_dp Parameter Jacobian in joint-model order
   */
  void updateParametrizationDerivative(Eigen::Ref<MatrixXs> dgamma_dp) const;

  /**
   * @brief Assemble the joint-model torque regressors
   *
   * Parameter columns follow joint id and insertion order and preserve each
   * concrete joint model's regressor convention.
   *
   * @param[out] dtau_dp Generalized-torque parameter regressor
   * @param[in] x Multibody state
   * @param[in] u Concatenated joint commands
   */
  void computeJointTorqueRegressor(Eigen::Ref<MatrixXs> dtau_dp,
                                   const Eigen::Ref<const VectorXs>& x,
                                   const Eigen::Ref<const VectorXs>& u) const;

  /** @brief Cast the state and shared joint-model stack to another scalar */
  template <typename NewScalar>
  ActuationModelMultibodyTpl<NewScalar> cast() const;

  /** @brief Print the multibody actuation model */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::u_lb_;
  using Base::u_ub_;

 private:
  struct ConstructorTag {};

  ActuationModelMultibodyTpl(
      std::shared_ptr<StateMultibody> state,
      const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joints,
      ConstructorTag);

  static std::vector<std::shared_ptr<JointDynamicsModelAbstract> >
  createDefaultIdentityModels(const std::shared_ptr<StateMultibody>& state);
  static std::size_t computeNu(
      const std::shared_ptr<StateMultibody>& state,
      const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joints);

  void initialize(
      const std::vector<std::shared_ptr<JointDynamicsModelAbstract> >& joints);
  void updateBounds();
  void updateTorqueTransform(
      const std::shared_ptr<ActuationDataMultibody>& data) const;
  std::shared_ptr<ActuationDataMultibody> castData(
      const std::shared_ptr<ActuationDataAbstract>& data) const;

  std::vector<std::vector<std::shared_ptr<JointDynamicsModelAbstract> > >
      joints_;
  std::vector<std::size_t> joint_control_dims_;
  std::vector<std::size_t> joint_u_offset_;
  std::size_t np_;
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/actuations/multibody.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActuationModelMultibodyTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActuationDataMultibodyTpl)

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_MULTIBODY_HPP_
