///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_IDENTITY_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_IDENTITY_HPP_

#include "crocoddyl/multibody/actuations/joint-dynamics-base.hpp"

namespace crocoddyl {

/**
 * @brief Identity joint dynamics \f$\tau_j=u_j\f$
 *
 * This model has no parameters or friction and supports arbitrary joint
 * velocity dimension with \f$n_u=n_v\f$.
 */
template <typename _Scalar>
class JointDynamicsModelIdentityTpl
    : public JointDynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(JointDynamicsModelBase, JointDynamicsModelIdentityTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef JointDynamicsModelAbstractTpl<Scalar> Base;
  typedef JointDynamicsDataAbstractTpl<Scalar> JointDynamicsDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize identity dynamics for a joint
   * @param[in] id Pinocchio joint index
   * @param[in] nq Joint configuration dimension
   * @param[in] nv Joint velocity, torque and command dimension
   */
  JointDynamicsModelIdentityTpl(const pinocchio::JointIndex id,
                                const std::size_t nq, const std::size_t nv)
      : Base(id, nq, nv, nv) {}
  virtual ~JointDynamicsModelIdentityTpl() = default;

  /** @brief Compute \f$\tau_j=u_j\f$ with zero friction */
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
    data->tau = u;
    data->friction.setZero();
  }

  /**
   * @brief Validate the constant identity derivatives prepared in createData()
   */
#ifndef NDEBUG
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
    assert_pretty(data->dtau_dq.isZero(), "dtau_dq has wrong value");
    assert_pretty(data->dtau_dv.isZero(), "dtau_dv has wrong value");
    assert_pretty(
        MatrixXs(data->dtau_du).isApprox(MatrixXs::Identity(nv_, nu_)),
        "dtau_du has wrong value");
  }
#else
  virtual void calcDiff(const std::shared_ptr<JointDynamicsDataAbstract>&,
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
  }
#endif

  /** @brief Apply the inverse identity map \f$u_j=\tau_j\f$ */
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
    data->u = tau;
    data->friction.setZero();
  }

  /** @brief Create data with identity torque and command maps */
  virtual std::shared_ptr<JointDynamicsDataAbstract> createData() override {
    std::shared_ptr<JointDynamicsDataAbstract> data =
        std::allocate_shared<JointDynamicsDataAbstract>(
            Eigen::aligned_allocator<JointDynamicsDataAbstract>(), this);
    data->dtau_du.setIdentity();
    data->Mtau.setIdentity();
    return data;
  }

  /** @brief Cast the model to a different scalar type */
  template <typename NewScalar>
  JointDynamicsModelIdentityTpl<NewScalar> cast() const {
    typedef JointDynamicsModelIdentityTpl<NewScalar> ReturnType;
    ReturnType ret(id_, nq_, nv_);
    return ret;
  }

  /** @brief Print the identity joint-dynamics model */
  virtual void print(std::ostream& os) const override {
    os << "JointDynamicsModelIdentity {id=" << id_ << ", nq=" << nq_
       << ", nv=" << nv_ << ", nu=" << nu_ << "}";
  }

 protected:
  using Base::id_;
  using Base::nq_;
  using Base::nu_;
  using Base::nv_;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::JointDynamicsModelIdentityTpl)

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_IDENTITY_HPP_
