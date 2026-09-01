///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_FRICTION_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_FRICTION_HPP_

#include <cmath>

#include "crocoddyl/multibody/actuations/joint-dynamics-base.hpp"

namespace crocoddyl {

/** @brief Supported differentiable friction parameterizations */
enum class JointFrictionType {
  Coulomb = 0,                         //!< Coulomb friction
  Viscous = 1,                         //!< Viscous friction
  Stribeck = 2,                        //!< Stribeck friction
  CoulombViscous = 3,                  //!< Coulomb and viscous friction
  CoulombStribeck = 4,                 //!< Coulomb and Stribeck friction
  ViscousStribeck = 5,                 //!< Viscous and Stribeck friction
  Full = 6,                            //!< All friction terms
  CoulombFixedSmoothing = 7,           //!< Coulomb with fixed smoothing
  StribeckFixedSmoothing = 8,          //!< Stribeck with fixed smoothing
  CoulombViscousFixedSmoothing = 9,    //!< Coulomb-viscous, fixed smoothing
  CoulombStribeckFixedSmoothing = 10,  //!< Coulomb-Stribeck, fixed smoothing
  ViscousStribeckFixedSmoothing = 11,  //!< Viscous-Stribeck, fixed smoothing
  FullFixedSmoothing = 12              //!< All terms with fixed smoothing
};

/** @brief Joint-friction data with cached hyperbolic-tangent terms */
template <typename _Scalar>
struct JointDynamicsDataFrictionTpl
    : public JointDynamicsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef JointDynamicsDataAbstractTpl<Scalar> Base;

  /**
   * @brief Allocate friction data and initialize its velocity caches
   * @param[in] model Friction model used for sizing; it is not retained
   */
  template <template <typename Scalar> class Model>
  explicit JointDynamicsDataFrictionTpl(Model<Scalar>* const model)
      : Base(model),
        tanh1v(Scalar(0.)),
        tanh2v(Scalar(0.)),
        tanh4v(Scalar(0.)) {}

  Scalar tanh1v;  //!< Cached first Stribeck hyperbolic-tangent term
  Scalar tanh2v;  //!< Cached second Stribeck hyperbolic-tangent term
  Scalar tanh4v;  //!< Cached Coulomb hyperbolic-tangent term
};

/**
 * @brief Differentiable one-DoF joint friction dynamics
 *
 * The selected mode combines Coulomb, viscous and Stribeck terms. Optimized
 * coefficients use the model parameterization, while fixed-smoothing modes
 * retain their smoothing coefficients outside \f$p\f$. Passive models have
 * \f$n_u=0\f$; active models map one command to one joint torque.
 */
template <typename _Scalar>
class JointDynamicsModelFrictionTpl
    : public JointDynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(JointDynamicsModelBase, JointDynamicsModelFrictionTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef JointDynamicsModelAbstractTpl<Scalar> Base;
  typedef JointDynamicsDataAbstractTpl<Scalar> JointDynamicsDataAbstract;
  typedef JointDynamicsDataFrictionTpl<Scalar> JointDynamicsDataFriction;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3s;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6s;

  /**
   * @brief Initialize active or passive friction dynamics
   *
   * Smoothing coefficients participate in \f$p\f$ for non-fixed modes.
   * Passive models have no command input and produce only friction torque.
   *
   * @param[in] id Pinocchio joint index
   * @param[in] nq Joint configuration dimension
   * @param[in] mu Unconstrained friction parameterization
   * @param[in] friction_type Friction terms and parameterization mode
   * @param[in] passive Whether the model has no active command
   */
  JointDynamicsModelFrictionTpl(
      const pinocchio::JointIndex id, const std::size_t nq,
      const Eigen::Ref<const VectorXs>& mu,
      const JointFrictionType friction_type = JointFrictionType::Full,
      const bool passive = false)
      : Base(id, nq, 1, passive ? 0 : 1),
        friction_type_(friction_type),
        np_(0),
        ns_(0),
        gamma_(Vector6s::Zero()),
        fixed_smoothing_(Vector3s::Zero()) {
    const VectorXs fixed_smoothing;
    initialize(mu, fixed_smoothing);
  }

  /**
   * @brief Initialize active friction dynamics with fixed smoothing
   *
   * @param[in] id Pinocchio joint index
   * @param[in] nq Joint configuration dimension
   * @param[in] mu Unconstrained optimizable parameterization
   * @param[in] friction_type Fixed-smoothing friction mode
   * @param[in] fixed_smoothing Fixed smoothing parameterization
   */
  JointDynamicsModelFrictionTpl(
      const pinocchio::JointIndex id, const std::size_t nq,
      const Eigen::Ref<const VectorXs>& mu,
      const JointFrictionType friction_type,
      const Eigen::Ref<const VectorXs>& fixed_smoothing)
      : JointDynamicsModelFrictionTpl(id, nq, mu, friction_type, false,
                                      fixed_smoothing) {}

  /**
   * @brief Initialize active or passive friction with fixed smoothing
   *
   * Fixed-smoothing entries are stored separately and excluded from \f$p\f$.
   *
   * @param[in] id Pinocchio joint index
   * @param[in] nq Joint configuration dimension
   * @param[in] mu Unconstrained optimizable parameterization
   * @param[in] friction_type Fixed-smoothing friction mode
   * @param[in] passive Whether the model has no active command
   * @param[in] fixed_smoothing Fixed smoothing parameterization
   */
  JointDynamicsModelFrictionTpl(
      const pinocchio::JointIndex id, const std::size_t nq,
      const Eigen::Ref<const VectorXs>& mu,
      const JointFrictionType friction_type, const bool passive,
      const Eigen::Ref<const VectorXs>& fixed_smoothing)
      : Base(id, nq, 1, passive ? 0 : 1),
        friction_type_(friction_type),
        np_(0),
        ns_(0),
        gamma_(Vector6s::Zero()),
        fixed_smoothing_(Vector3s::Zero()) {
    initialize(mu, fixed_smoothing);
  }
  virtual ~JointDynamicsModelFrictionTpl() = default;

  /**
   * @brief Compute \f$\tau=u-\tau_f(v)\f$ and cache friction terms
   *
   * For passive models the command contribution is zero.
   */
  virtual void calc(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& q,
                    const Eigen::Ref<const VectorXs>& v,
                    const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(q.size()) != nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    const std::shared_ptr<JointDynamicsDataFriction> d = castData(data);
    updateFrictionTerms(d, v[0]);
    const Scalar control = nu_ == 0 ? Scalar(0.) : u[0];
    d->tau[0] = control - d->friction[0];
  }

  /**
   * @brief Compute friction derivatives from terms cached by calc()
   *
   * calc() must be called first with the same data and inputs.
   */
  virtual void calcDiff(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& q,
                        const Eigen::Ref<const VectorXs>& v,
                        const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(q.size()) != nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    const std::shared_ptr<JointDynamicsDataFriction> d = castData(data);
    d->dtau_dq.setZero();
    d->dtau_dv.setZero();
    d->dtau_dv(0, 0) =
        -gamma_[0] * gamma_[1] * (Scalar(1.) - d->tanh1v * d->tanh1v) +
        gamma_[0] * gamma_[2] * (Scalar(1.) - d->tanh2v * d->tanh2v) -
        gamma_[3] * gamma_[4] * (Scalar(1.) - d->tanh4v * d->tanh4v) -
        gamma_[5];
    if (nu_ != 0) {
      d->dtau_du.setIdentity();
    } else {
      d->dtau_du.setZero();
    }
  }

  /**
   * @brief Compute the command required to balance friction and desired torque
   *
   * Passive models return an empty command.
   */
  virtual void commands(const std::shared_ptr<JointDynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& q,
                        const Eigen::Ref<const VectorXs>& v,
                        const Eigen::Ref<const VectorXs>& tau) override {
    if (static_cast<std::size_t>(q.size()) != nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    if (static_cast<std::size_t>(tau.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "tau has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    const std::shared_ptr<JointDynamicsDataFriction> d = castData(data);
    updateFrictionTerms(d, v[0]);
    if (nu_ != 0) {
      d->u[0] = tau[0] + d->friction[0];
    } else {
      d->u.setZero();
    }
  }

  /**
   * @brief Compute the positive-friction regressor
   *
   * By the established friction convention this is
   * \f$\partial\tau_f/\partial p=-\partial\tau/\partial p\f$.
   *
   * @param[out] joint_dtau_dp Positive-friction parameter regressor
   * @param[in] q Joint configuration
   * @param[in] v Joint velocity
   * @param[in] u Joint command, empty for passive models
   */
  void computeJointTorqueRegressor(
      Eigen::Ref<MatrixXs> joint_dtau_dp, const Eigen::Ref<const VectorXs>& q,
      const Eigen::Ref<const VectorXs>& v,
      const Eigen::Ref<const VectorXs>& u) const override {
    if (static_cast<std::size_t>(q.size()) != nq_) {
      throw_pretty(
          "Invalid argument: " << "q has wrong dimension (it should be " +
                                      std::to_string(nq_) + ")");
    }
    if (static_cast<std::size_t>(v.size()) != nv_) {
      throw_pretty(
          "Invalid argument: " << "v has wrong dimension (it should be " +
                                      std::to_string(nv_) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    if (static_cast<std::size_t>(joint_dtau_dp.rows()) != nv_ ||
        static_cast<std::size_t>(joint_dtau_dp.cols()) != np_) {
      throw_pretty("Invalid argument: joint_dtau_dp has wrong dimensions");
    }

    joint_dtau_dp.setZero();
    const Scalar velocity = v[0];
    using std::tanh;
    const Scalar tanh1v = tanh(gamma_[1] * velocity);
    const Scalar tanh2v = tanh(gamma_[2] * velocity);
    const Scalar tanh4v = tanh(gamma_[4] * velocity);
    const Scalar sech21v = Scalar(1.) - tanh1v * tanh1v;
    const Scalar sech22v = Scalar(1.) - tanh2v * tanh2v;
    const Scalar sech24v = Scalar(1.) - tanh4v * tanh4v;

    switch (friction_type_) {
      case JointFrictionType::Coulomb:
        joint_dtau_dp(0, 0) = gamma_[3] * tanh4v;
        joint_dtau_dp(0, 1) = gamma_[3] * sech24v * gamma_[4] * velocity;
        break;
      case JointFrictionType::CoulombFixedSmoothing:
        joint_dtau_dp(0, 0) = gamma_[3] * tanh4v;
        break;
      case JointFrictionType::Viscous:
        joint_dtau_dp(0, 0) = gamma_[5] * velocity;
        break;
      case JointFrictionType::Stribeck:
        joint_dtau_dp(0, 0) = sech21v * velocity;
        joint_dtau_dp(0, 1) = -sech22v * velocity;
        break;
      case JointFrictionType::StribeckFixedSmoothing:
        break;
      case JointFrictionType::CoulombViscous:
        joint_dtau_dp(0, 0) = gamma_[3] * tanh4v;
        joint_dtau_dp(0, 1) = gamma_[3] * sech24v * gamma_[4] * velocity;
        joint_dtau_dp(0, 2) = gamma_[5] * velocity;
        break;
      case JointFrictionType::CoulombViscousFixedSmoothing:
        joint_dtau_dp(0, 0) = gamma_[3] * tanh4v;
        joint_dtau_dp(0, 1) = gamma_[5] * velocity;
        break;
      case JointFrictionType::CoulombStribeck:
        joint_dtau_dp(0, 0) = sech21v * velocity;
        joint_dtau_dp(0, 1) = -sech22v * velocity;
        joint_dtau_dp(0, 2) = tanh4v;
        joint_dtau_dp(0, 3) = gamma_[3] * sech24v * velocity;
        break;
      case JointFrictionType::CoulombStribeckFixedSmoothing:
        joint_dtau_dp(0, 0) = tanh4v;
        break;
      case JointFrictionType::ViscousStribeck:
        joint_dtau_dp(0, 0) = sech21v * velocity;
        joint_dtau_dp(0, 1) = -sech22v * velocity;
        joint_dtau_dp(0, 2) = gamma_[5] * velocity;
        break;
      case JointFrictionType::ViscousStribeckFixedSmoothing:
        joint_dtau_dp(0, 0) = gamma_[5] * velocity;
        break;
      case JointFrictionType::Full:
        joint_dtau_dp(0, 0) = tanh1v - tanh2v;
        joint_dtau_dp(0, 1) = gamma_[0] * sech21v * velocity;
        joint_dtau_dp(0, 2) = -gamma_[0] * sech22v * velocity;
        joint_dtau_dp(0, 3) = tanh4v;
        joint_dtau_dp(0, 4) = gamma_[3] * sech24v * velocity;
        joint_dtau_dp(0, 5) = gamma_[5] * velocity;
        break;
      case JointFrictionType::FullFixedSmoothing:
        joint_dtau_dp(0, 0) = tanh1v - tanh2v;
        joint_dtau_dp(0, 1) = tanh4v;
        joint_dtau_dp(0, 2) = gamma_[5] * velocity;
        break;
      default:
        throw_pretty("Invalid argument: unsupported friction type");
        break;
    }
  }

  /**
   * @brief Set physical friction coefficients from their parameterization
   * @param[in] p Unconstrained parameter vector of dimension get_np()
   */
  void set_parameters(const Eigen::Ref<const VectorXs>& p) override {
    if (static_cast<std::size_t>(p.size()) != np_) {
      throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                   std::to_string(np_) + ")");
    }
    gamma_.setZero();
    using std::exp;
    switch (friction_type_) {
      case JointFrictionType::Coulomb:
        gamma_[3] = exp(p[0]);
        gamma_[4] = exp(p[1]);
        break;
      case JointFrictionType::CoulombFixedSmoothing:
        gamma_[3] = exp(p[0]);
        gamma_[4] = exp(fixed_smoothing_[0]);
        break;
      case JointFrictionType::Viscous:
        gamma_[5] = exp(p[0]);
        break;
      case JointFrictionType::Stribeck:
        gamma_[0] = Scalar(1.);
        gamma_[1] = p[0];
        gamma_[2] = p[1];
        break;
      case JointFrictionType::StribeckFixedSmoothing:
        gamma_[0] = Scalar(1.);
        gamma_[1] = fixed_smoothing_[0];
        gamma_[2] = fixed_smoothing_[1];
        break;
      case JointFrictionType::CoulombViscous:
        gamma_[3] = exp(p[0]);
        gamma_[4] = exp(p[1]);
        gamma_[5] = exp(p[2]);
        break;
      case JointFrictionType::CoulombViscousFixedSmoothing:
        gamma_[3] = exp(p[0]);
        gamma_[4] = exp(fixed_smoothing_[0]);
        gamma_[5] = exp(p[1]);
        break;
      case JointFrictionType::CoulombStribeck:
        gamma_[0] = Scalar(1.);
        gamma_[1] = p[0];
        gamma_[2] = p[1];
        gamma_[3] = p[2];
        gamma_[4] = p[3];
        break;
      case JointFrictionType::CoulombStribeckFixedSmoothing:
        gamma_[0] = Scalar(1.);
        gamma_[1] = fixed_smoothing_[0];
        gamma_[2] = fixed_smoothing_[1];
        gamma_[3] = p[0];
        gamma_[4] = fixed_smoothing_[2];
        break;
      case JointFrictionType::ViscousStribeck:
        gamma_[0] = Scalar(1.);
        gamma_[1] = p[0];
        gamma_[2] = p[1];
        gamma_[5] = exp(p[2]);
        break;
      case JointFrictionType::ViscousStribeckFixedSmoothing:
        gamma_[0] = Scalar(1.);
        gamma_[1] = fixed_smoothing_[0];
        gamma_[2] = fixed_smoothing_[1];
        gamma_[5] = exp(p[0]);
        break;
      case JointFrictionType::Full:
        gamma_.template head<5>() = p.template head<5>();
        gamma_[5] = exp(p[5]);
        break;
      case JointFrictionType::FullFixedSmoothing:
        gamma_[0] = p[0];
        gamma_[1] = fixed_smoothing_[0];
        gamma_[2] = fixed_smoothing_[1];
        gamma_[3] = p[1];
        gamma_[4] = fixed_smoothing_[2];
        gamma_[5] = exp(p[2]);
        break;
      default:
        throw_pretty("Invalid argument: unsupported friction type");
        break;
    }
  }

  /** @brief Return the physical optimizable friction coefficients */
  VectorXs get_parameters() const override {
    VectorXs parameters(np_);
    switch (friction_type_) {
      case JointFrictionType::Coulomb:
        parameters[0] = gamma_[3];
        parameters[1] = gamma_[4];
        break;
      case JointFrictionType::CoulombFixedSmoothing:
        parameters[0] = gamma_[3];
        break;
      case JointFrictionType::Viscous:
        parameters[0] = gamma_[5];
        break;
      case JointFrictionType::Stribeck:
        parameters[0] = gamma_[1];
        parameters[1] = gamma_[2];
        break;
      case JointFrictionType::StribeckFixedSmoothing:
        break;
      case JointFrictionType::CoulombViscous:
        parameters[0] = gamma_[3];
        parameters[1] = gamma_[4];
        parameters[2] = gamma_[5];
        break;
      case JointFrictionType::CoulombViscousFixedSmoothing:
        parameters[0] = gamma_[3];
        parameters[1] = gamma_[5];
        break;
      case JointFrictionType::CoulombStribeck:
        parameters[0] = gamma_[1];
        parameters[1] = gamma_[2];
        parameters[2] = gamma_[3];
        parameters[3] = gamma_[4];
        break;
      case JointFrictionType::CoulombStribeckFixedSmoothing:
        parameters[0] = gamma_[3];
        break;
      case JointFrictionType::ViscousStribeck:
        parameters[0] = gamma_[1];
        parameters[1] = gamma_[2];
        parameters[2] = gamma_[5];
        break;
      case JointFrictionType::ViscousStribeckFixedSmoothing:
        parameters[0] = gamma_[5];
        break;
      case JointFrictionType::Full:
        parameters = gamma_;
        break;
      case JointFrictionType::FullFixedSmoothing:
        parameters[0] = gamma_[0];
        parameters[1] = gamma_[3];
        parameters[2] = gamma_[5];
        break;
      default:
        throw_pretty("Invalid argument: unsupported friction type");
        break;
    }
    return parameters;
  }

  /** @brief Return the unconstrained optimizable parameterization */
  VectorXs get_parametrization() const override {
    return to_parametrization(get_parameters());
  }

  /**
   * @brief Map unconstrained parameters to physical coefficients
   * @param[in] mu Unconstrained friction parameterization
   */
  VectorXs from_parametrization(const Eigen::Ref<const VectorXs>& mu) const {
    if (static_cast<std::size_t>(mu.size()) != np_) {
      throw_pretty("Invalid argument: mu has wrong dimension (it should be " +
                   std::to_string(np_) + ")");
    }
    VectorXs gamma(np_);
    using std::exp;
    switch (friction_type_) {
      case JointFrictionType::Coulomb:
        gamma[0] = exp(mu[0]);
        gamma[1] = exp(mu[1]);
        break;
      case JointFrictionType::CoulombFixedSmoothing:
        gamma[0] = exp(mu[0]);
        break;
      case JointFrictionType::Viscous:
        gamma[0] = exp(mu[0]);
        break;
      case JointFrictionType::Stribeck:
        gamma[0] = mu[0];
        gamma[1] = mu[1];
        break;
      case JointFrictionType::StribeckFixedSmoothing:
        break;
      case JointFrictionType::CoulombViscous:
        gamma[0] = exp(mu[0]);
        gamma[1] = exp(mu[1]);
        gamma[2] = exp(mu[2]);
        break;
      case JointFrictionType::CoulombViscousFixedSmoothing:
        gamma[0] = exp(mu[0]);
        gamma[1] = exp(mu[1]);
        break;
      case JointFrictionType::CoulombStribeck:
        gamma[0] = mu[0];
        gamma[1] = mu[1];
        gamma[2] = mu[2];
        gamma[3] = mu[3];
        break;
      case JointFrictionType::CoulombStribeckFixedSmoothing:
        gamma[0] = mu[0];
        break;
      case JointFrictionType::ViscousStribeck:
        gamma[0] = mu[0];
        gamma[1] = mu[1];
        gamma[2] = exp(mu[2]);
        break;
      case JointFrictionType::ViscousStribeckFixedSmoothing:
        gamma[0] = exp(mu[0]);
        break;
      case JointFrictionType::Full:
        gamma.template head<5>() = mu.template head<5>();
        gamma[5] = exp(mu[5]);
        break;
      case JointFrictionType::FullFixedSmoothing:
        gamma[0] = mu[0];
        gamma[1] = mu[1];
        gamma[2] = exp(mu[2]);
        break;
      default:
        throw_pretty("Invalid argument: unsupported friction type");
        break;
    }
    return gamma;
  }

  /**
   * @brief Map physical coefficients to unconstrained parameters
   * @param[in] gamma Physical optimizable friction coefficients
   */
  VectorXs to_parametrization(const Eigen::Ref<const VectorXs>& gamma) const {
    if (static_cast<std::size_t>(gamma.size()) != np_) {
      throw_pretty(
          "Invalid argument: gamma has wrong dimension (it should be " +
          std::to_string(np_) + ")");
    }
    VectorXs mu(np_);
    using std::log;
    switch (friction_type_) {
      case JointFrictionType::Coulomb:
        mu[0] = log(gamma[0]);
        mu[1] = log(gamma[1]);
        break;
      case JointFrictionType::CoulombFixedSmoothing:
        mu[0] = log(gamma[0]);
        break;
      case JointFrictionType::Viscous:
        mu[0] = log(gamma[0]);
        break;
      case JointFrictionType::Stribeck:
        mu[0] = gamma[0];
        mu[1] = gamma[1];
        break;
      case JointFrictionType::StribeckFixedSmoothing:
        break;
      case JointFrictionType::CoulombViscous:
        mu[0] = log(gamma[0]);
        mu[1] = log(gamma[1]);
        mu[2] = log(gamma[2]);
        break;
      case JointFrictionType::CoulombViscousFixedSmoothing:
        mu[0] = log(gamma[0]);
        mu[1] = log(gamma[1]);
        break;
      case JointFrictionType::CoulombStribeck:
        mu[0] = gamma[0];
        mu[1] = gamma[1];
        mu[2] = gamma[2];
        mu[3] = gamma[3];
        break;
      case JointFrictionType::CoulombStribeckFixedSmoothing:
        mu[0] = gamma[0];
        break;
      case JointFrictionType::ViscousStribeck:
        mu[0] = gamma[0];
        mu[1] = gamma[1];
        mu[2] = log(gamma[2]);
        break;
      case JointFrictionType::ViscousStribeckFixedSmoothing:
        mu[0] = log(gamma[0]);
        break;
      case JointFrictionType::Full:
        mu.template head<5>() = gamma.template head<5>();
        mu[5] = log(gamma[5]);
        break;
      case JointFrictionType::FullFixedSmoothing:
        mu[0] = gamma[0];
        mu[1] = gamma[1];
        mu[2] = log(gamma[2]);
        break;
      default:
        throw_pretty("Invalid argument: unsupported friction type");
        break;
    }
    return mu;
  }

  /**
   * @brief Compute the physical-parameter Jacobian
   * @param[out] dgamma_dp Jacobian of physical coefficients with respect to
   * the unconstrained parameterization
   */
  void updateParametrizationDerivative(
      Eigen::Ref<MatrixXs> dgamma_dp) const override {
    if (static_cast<std::size_t>(dgamma_dp.rows()) != np_ ||
        static_cast<std::size_t>(dgamma_dp.cols()) != np_) {
      throw_pretty("Invalid argument: dgamma_dp has wrong dimensions");
    }
    dgamma_dp.setZero();
    dgamma_dp.diagonal().setOnes();
    if (friction_type_ == JointFrictionType::Viscous ||
        friction_type_ == JointFrictionType::Full ||
        friction_type_ == JointFrictionType::ViscousStribeck ||
        friction_type_ == JointFrictionType::ViscousStribeckFixedSmoothing) {
      dgamma_dp(np_ - 1, np_ - 1) = get_parameters()[np_ - 1];
    } else if (friction_type_ == JointFrictionType::CoulombViscous ||
               friction_type_ == JointFrictionType::Coulomb ||
               friction_type_ ==
                   JointFrictionType::CoulombViscousFixedSmoothing ||
               friction_type_ == JointFrictionType::CoulombFixedSmoothing) {
      dgamma_dp.diagonal() = get_parameters();
    } else if (friction_type_ == JointFrictionType::FullFixedSmoothing) {
      dgamma_dp(np_ - 1, np_ - 1) = get_parameters()[np_ - 1];
    }
  }

  /** @brief Create specialized data with active command maps when applicable */
  virtual std::shared_ptr<JointDynamicsDataAbstract> createData() override {
    std::shared_ptr<JointDynamicsDataFriction> data =
        std::allocate_shared<JointDynamicsDataFriction>(
            Eigen::aligned_allocator<JointDynamicsDataFriction>(), this);
    if (nu_ != 0) {
      data->dtau_du.setIdentity();
      data->Mtau.setIdentity();
    }
    return data;
  }

  /** @brief Return the selected friction mode */
  JointFrictionType get_friction_type() const { return friction_type_; }

  /** @brief Return the number of optimizable friction parameters */
  std::size_t get_np() const override { return np_; }

  /** @brief Return fixed smoothing in its stored parameterization */
  VectorXs get_fixed_smoothing_parametrization() const {
    VectorXs smoothing(ns_);
    for (std::size_t i = 0; i < ns_; ++i) {
      smoothing[static_cast<Eigen::Index>(i)] =
          fixed_smoothing_[static_cast<Eigen::Index>(i)];
    }
    return smoothing;
  }

  /** @brief Return whether the model is passive and has no command input */
  bool get_passive() const { return nu_ == 0; }

  /** @brief Cast the complete friction model to a different scalar type */
  template <typename NewScalar>
  JointDynamicsModelFrictionTpl<NewScalar> cast() const {
    typedef JointDynamicsModelFrictionTpl<NewScalar> ReturnType;
    ReturnType ret(
        id_, nq_, get_parametrization().template cast<NewScalar>(),
        friction_type_, get_passive(),
        get_fixed_smoothing_parametrization().template cast<NewScalar>());
    return ret;
  }

  /** @brief Print the friction joint-dynamics model */
  virtual void print(std::ostream& os) const override {
    os << "JointDynamicsModelFriction {id=" << id_ << ", nq=" << nq_
       << ", nv=" << nv_ << ", nu=" << nu_ << ", np=" << np_ << "}";
  }

 protected:
  using Base::id_;
  using Base::nq_;
  using Base::nu_;
  using Base::nv_;

 private:
  void configureDimensions() {
    switch (friction_type_) {
      case JointFrictionType::Coulomb:
        np_ = 2;
        ns_ = 0;
        break;
      case JointFrictionType::CoulombFixedSmoothing:
        np_ = 1;
        ns_ = 1;
        break;
      case JointFrictionType::Viscous:
        np_ = 1;
        ns_ = 0;
        break;
      case JointFrictionType::Stribeck:
        np_ = 2;
        ns_ = 0;
        break;
      case JointFrictionType::StribeckFixedSmoothing:
        np_ = 0;
        ns_ = 2;
        break;
      case JointFrictionType::CoulombViscous:
        np_ = 3;
        ns_ = 0;
        break;
      case JointFrictionType::CoulombViscousFixedSmoothing:
        np_ = 2;
        ns_ = 1;
        break;
      case JointFrictionType::CoulombStribeck:
        np_ = 4;
        ns_ = 0;
        break;
      case JointFrictionType::CoulombStribeckFixedSmoothing:
        np_ = 1;
        ns_ = 3;
        break;
      case JointFrictionType::ViscousStribeck:
        np_ = 3;
        ns_ = 0;
        break;
      case JointFrictionType::ViscousStribeckFixedSmoothing:
        np_ = 1;
        ns_ = 2;
        break;
      case JointFrictionType::Full:
        np_ = 6;
        ns_ = 0;
        break;
      case JointFrictionType::FullFixedSmoothing:
        np_ = 3;
        ns_ = 3;
        break;
      default:
        throw_pretty("Invalid argument: unsupported friction type");
        break;
    }
  }

  void initialize(const Eigen::Ref<const VectorXs>& mu,
                  const Eigen::Ref<const VectorXs>& fixed_smoothing) {
    configureDimensions();
    if (static_cast<std::size_t>(mu.size()) != np_) {
      throw_pretty("Invalid argument: mu has wrong dimension (it should be " +
                   std::to_string(np_) + ")");
    }
    if (static_cast<std::size_t>(fixed_smoothing.size()) != ns_) {
      throw_pretty(
          "Invalid argument: fixed_smoothing has wrong dimension (it should "
          "be " +
          std::to_string(ns_) + ")");
    }
    fixed_smoothing_.setZero();
    for (std::size_t i = 0; i < ns_; ++i) {
      fixed_smoothing_[static_cast<Eigen::Index>(i)] =
          fixed_smoothing[static_cast<Eigen::Index>(i)];
    }
    set_parameters(mu);
  }

  std::shared_ptr<JointDynamicsDataFriction> castData(
      const std::shared_ptr<JointDynamicsDataAbstract>& data) const {
    std::shared_ptr<JointDynamicsDataFriction> d =
        std::dynamic_pointer_cast<JointDynamicsDataFriction>(data);
    if (d == nullptr) {
      throw_pretty("Invalid argument: data is not a JointDynamicsDataFriction");
    }
    return d;
  }

  void updateFrictionTerms(
      const std::shared_ptr<JointDynamicsDataFriction>& data,
      const Scalar velocity) const {
    using std::tanh;
    data->tanh1v = tanh(gamma_[1] * velocity);
    data->tanh2v = tanh(gamma_[2] * velocity);
    data->tanh4v = tanh(gamma_[4] * velocity);
    data->friction[0] = gamma_[0] * data->tanh1v - gamma_[0] * data->tanh2v +
                        gamma_[3] * data->tanh4v + gamma_[5] * velocity;
  }

  JointFrictionType friction_type_;
  std::size_t np_;
  std::size_t ns_;
  Vector6s gamma_;
  Vector3s fixed_smoothing_;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::JointDynamicsDataFrictionTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::JointDynamicsModelFrictionTpl)

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_FRICTION_HPP_
