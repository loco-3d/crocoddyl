///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_DYNAMICS_BASE_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_DYNAMICS_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

/** @brief Common type-erased base for joint-dynamics models */
class JointDynamicsModelBase {
 public:
  virtual ~JointDynamicsModelBase() = default;

  CROCODDYL_BASE_CAST(JointDynamicsModelBase, JointDynamicsModelAbstractTpl)
};

/**
 * @brief Abstract actuation dynamics for one Pinocchio joint
 *
 * A joint model maps configuration \f$q_j\in\mathbb{R}^{n_q}\f$, velocity
 * \f$v_j\in\mathbb{R}^{n_v}\f$ and command \f$u_j\in\mathbb{R}^{n_u}\f$ to
 * generalized joint torque and friction. It also defines the inverse command
 * map, derivatives, an optional \f$n_p\f$-parameter representation and its
 * torque regressor. Each model owns no runtime data; createData() returns an
 * independent payload of matching dimensions. calc() must be called before
 * calcDiff() for the same data and inputs, as calcDiff() uses values prepared
 * by calc().
 */
template <typename _Scalar>
class JointDynamicsModelAbstractTpl : public JointDynamicsModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef JointDynamicsDataAbstractTpl<Scalar> JointDynamicsDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize a joint-dynamics model
   *
   * @param[in] id Pinocchio joint index
   * @param[in] nq Joint configuration dimension
   * @param[in] nv Joint velocity and torque dimension
   * @param[in] nu Joint command dimension
   */
  JointDynamicsModelAbstractTpl(const pinocchio::JointIndex id,
                                const std::size_t nq, const std::size_t nv,
                                const std::size_t nu = 0);
  virtual ~JointDynamicsModelAbstractTpl() = default;

  /**
   * @brief Compute joint torque and cache quantities required by calcDiff()
   *
   * @param[in,out] data Joint-dynamics data
   * @param[in] q Joint configuration
   * @param[in] v Joint velocity
   * @param[in] u Joint command
   */
  virtual void calc(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& q,
                    const Eigen::Ref<const VectorXs>& v,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the joint-torque derivatives
   *
   * calc() must be called first with the same data and inputs.
   *
   * @param[in,out] data Joint-dynamics data
   * @param[in] q Joint configuration
   * @param[in] v Joint velocity
   * @param[in] u Joint command
   */
  virtual void calcDiff(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& q,
                        const Eigen::Ref<const VectorXs>& v,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute commands that realize a desired joint torque
   *
   * @param[in,out] data Joint-dynamics data containing the resulting command
   * @param[in] q Joint configuration
   * @param[in] v Joint velocity
   * @param[in] tau Desired joint torque
   */
  virtual void commands(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& q,
                        const Eigen::Ref<const VectorXs>& v,
                        const Eigen::Ref<const VectorXs>& tau) = 0;

  /** @brief Return the number of optimizable parameters */
  virtual std::size_t get_np() const;

  /**
   * @brief Update the model from its parameterization
   * @param[in] p Unconstrained parameter vector of dimension get_np()
   */
  virtual void set_parameters(const Eigen::Ref<const VectorXs>& p);

  /** @brief Return the physical parameters */
  virtual VectorXs get_parameters() const;

  /** @brief Return the unconstrained parameterization */
  virtual VectorXs get_parametrization() const;

  /**
   * @brief Compute the physical-parameter Jacobian
   * @param[out] dgamma_dp Jacobian of physical parameters with respect to the
   * unconstrained parameterization
   */
  virtual void updateParametrizationDerivative(
      Eigen::Ref<MatrixXs> dgamma_dp) const;

  /**
   * @brief Compute the model-specific joint-torque parameter regressor
   *
   * The sign follows the concrete model contract: friction models return the
   * positive-friction derivative \f$-\partial\tau/\partial p\f$, while direct
   * torque mappings return \f$\partial\tau/\partial p\f$.
   *
   * @param[out] joint_dtau_dp Joint-torque regressor
   * @param[in] q Joint configuration
   * @param[in] v Joint velocity
   * @param[in] u Joint command
   */
  virtual void computeJointTorqueRegressor(
      Eigen::Ref<MatrixXs> joint_dtau_dp, const Eigen::Ref<const VectorXs>& q,
      const Eigen::Ref<const VectorXs>& v,
      const Eigen::Ref<const VectorXs>& u) const;

  /** @brief Create an independent data payload for this model */
  virtual std::shared_ptr<JointDynamicsDataAbstract> createData();

  /** @brief Return the Pinocchio joint index */
  pinocchio::JointIndex get_id() const;

  /** @brief Return the joint configuration dimension */
  std::size_t get_nq() const;

  /** @brief Return the joint velocity and torque dimension */
  std::size_t get_nv() const;

  /** @brief Return the joint command dimension */
  std::size_t get_nu() const;

  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os, const JointDynamicsModelAbstractTpl<Scalar>& model);

  /** @brief Print the model information */
  virtual void print(std::ostream& os) const;

 protected:
  pinocchio::JointIndex id_;  //!< Pinocchio joint index
  std::size_t nq_;            //!< Joint configuration dimension
  std::size_t nv_;            //!< Joint velocity and torque dimension
  std::size_t nu_;            //!< Joint command dimension
  JointDynamicsModelAbstractTpl() : id_(0), nq_(0), nv_(0), nu_(0) {}
};

/**
 * @brief Runtime data shared by joint-dynamics models
 *
 * Stores friction and torque in \f$\mathbb{R}^{n_v}\f$, commands in
 * \f$\mathbb{R}^{n_u}\f$, derivatives \f$d\tau/dq,d\tau/dv\in
 * \mathbb{R}^{n_v\times n_v}\f$, \f$d\tau/du\in
 * \mathbb{R}^{n_v\times n_u}\f$, and the inverse torque map \f$M_\tau\in
 * \mathbb{R}^{n_u\times n_v}\f$.
 */
template <typename _Scalar>
struct JointDynamicsDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Allocate and initialize data from a joint model
   *
   * The model is used only to determine dimensions and is not retained.
   *
   * @param[in] model Joint-dynamics model
   */
  template <template <typename Scalar> class Model>
  explicit JointDynamicsDataAbstractTpl(Model<Scalar>* const model)
      : friction(model == nullptr ? 0 : model->get_nv()),
        tau(model == nullptr ? 0 : model->get_nv()),
        u(model == nullptr ? 0 : model->get_nu()),
        dtau_dq(model == nullptr ? 0 : model->get_nv(),
                model == nullptr ? 0 : model->get_nv()),
        dtau_dv(model == nullptr ? 0 : model->get_nv(),
                model == nullptr ? 0 : model->get_nv()),
        dtau_du(model == nullptr ? 0 : model->get_nv(),
                model == nullptr ? 0 : model->get_nu()),
        Mtau(model == nullptr ? 0 : model->get_nu(),
             model == nullptr ? 0 : model->get_nv()) {
    if (model == nullptr) {
      throw_pretty("Invalid argument: model is null");
    }
    friction.setZero();
    tau.setZero();
    u.setZero();
    dtau_dq.setZero();
    dtau_dv.setZero();
    dtau_du.setZero();
    Mtau.setZero();
  }
  virtual ~JointDynamicsDataAbstractTpl() = default;

  VectorXs friction;  //!< Friction torque
  VectorXs tau;       //!< Generalized joint torque
  VectorXs u;         //!< Joint command
  MatrixXs dtau_dq;   //!< Joint-torque derivative with respect to configuration
  MatrixXs dtau_dv;   //!< Joint-torque derivative with respect to velocity
  MatrixXs dtau_du;   //!< Joint-torque derivative with respect to command
  MatrixXs Mtau;      //!< Command map from desired joint torque
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/actuations/joint-dynamics-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::JointDynamicsModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::JointDynamicsDataAbstractTpl)

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_DYNAMICS_BASE_HPP_
