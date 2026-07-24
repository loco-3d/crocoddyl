///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_THRUSTER_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_THRUSTER_HPP_

#include "crocoddyl/multibody/actuations/floating-base-thrusters.hpp"
#include "crocoddyl/multibody/actuations/joint-dynamics-base.hpp"

namespace crocoddyl {

/**
 * @brief Free-flyer joint dynamics driven by a set of thrusters
 *
 * The spatial wrench is \f$\tau_j=W_{\mathrm{thrust}}u\f$ and commands are
 * recovered with its pseudoinverse. The parameter vector contains one torque
 * coefficient per thruster. Thruster pose, direction and bounds remain part
 * of the shared thruster model.
 */
template <typename _Scalar>
class JointDynamicsModelThrusterTpl
    : public JointDynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(JointDynamicsModelBase, JointDynamicsModelThrusterTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef JointDynamicsModelAbstractTpl<Scalar> Base;
  typedef JointDynamicsDataAbstractTpl<Scalar> JointDynamicsDataAbstract;
  typedef ThrusterTpl<Scalar> Thruster;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize free-flyer dynamics from a set of thrusters
   *
   * The model stores its own thruster descriptors and rebuilds the wrench and
   * pseudoinverse command maps whenever their torque coefficients change.
   *
   * @param[in] thrusters Thruster descriptors in command order
   */
  explicit JointDynamicsModelThrusterTpl(const std::vector<Thruster>& thrusters)
      : Base(1, 7, 6, thrusters.size()),
        thrusters_(thrusters),
        n_thrusters_(thrusters.size()),
        W_thrust_(6, n_thrusters_),
        Mtau_(n_thrusters_, 6) {
    set_thrusters(thrusters_);
  }
  virtual ~JointDynamicsModelThrusterTpl() = default;

  /**
   * @brief Compute the spatial wrench and prepare the torque/command mappings
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
    data->dtau_du = W_thrust_;
    data->Mtau = Mtau_;
    data->tau.noalias() = data->dtau_du * u;
    data->friction.setZero();
  }

  /**
   * @brief Compute the constant derivatives using mappings prepared by calc()
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
    data->dtau_dq.setZero();
    data->dtau_dv.setZero();
#ifndef NDEBUG
    assert_pretty(MatrixXs(data->dtau_du).isApprox(W_thrust_),
                  "dtau_du has wrong value");
    assert_pretty(MatrixXs(data->Mtau).isApprox(Mtau_), "Mtau has wrong value");
#endif
  }

  /** @brief Compute minimum-norm thruster commands for a spatial wrench */
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
    data->dtau_du = W_thrust_;
    data->Mtau = Mtau_;
    data->u.noalias() = data->Mtau * tau;
    data->friction.setZero();
  }

  /**
   * @brief Compute \f$\partial\tau/\partial p\f$ for thruster torque
   * coefficients
   * @param[out] joint_dtau_dp Spatial-wrench parameter regressor
   * @param[in] q Free-flyer configuration
   * @param[in] v Free-flyer velocity
   * @param[in] u Thruster commands
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
        static_cast<std::size_t>(joint_dtau_dp.cols()) != n_thrusters_) {
      throw_pretty("Invalid argument: joint_dtau_dp has wrong dimensions");
    }

    joint_dtau_dp.setZero();
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
      const Vector3s f_z = thrusters_[i].pose.rotation() * Vector3s::UnitZ();
      const Scalar sign = thrusters_[i].type == CW ? Scalar(1.) : Scalar(-1.);
      joint_dtau_dp.template middleRows<3>(3).col(i).noalias() =
          sign * u[i] * f_z;
    }
  }

  /** @brief Create data initialized with the current wrench mappings */
  virtual std::shared_ptr<JointDynamicsDataAbstract> createData() override {
    std::shared_ptr<JointDynamicsDataAbstract> data =
        std::allocate_shared<JointDynamicsDataAbstract>(
            Eigen::aligned_allocator<JointDynamicsDataAbstract>(), this);
    data->dtau_du = W_thrust_;
    data->Mtau = Mtau_;
    return data;
  }

  /** @brief Return the copied thruster descriptors in command order */
  const std::vector<Thruster>& get_thrusters() const { return thrusters_; }

  /** @brief Return the number of thrusters */
  std::size_t get_nthrusters() const { return n_thrusters_; }

  /** @brief Return the number of torque-coefficient parameters */
  std::size_t get_np() const override { return n_thrusters_; }

  /** @brief Return the spatial-wrench mapping */
  const MatrixXs& get_Wthrust() const { return W_thrust_; }

  /**
   * @brief Replace the thruster descriptors without changing their count
   * @param[in] thrusters Thruster descriptors in command order
   */
  void set_thrusters(const std::vector<Thruster>& thrusters) {
    if (static_cast<std::size_t>(thrusters.size()) != n_thrusters_) {
      throw_pretty("Invalid argument: "
                   << "the number of thrusters is wrong (it should be " +
                          std::to_string(n_thrusters_) + ")");
    }
    thrusters_ = thrusters;
    updateMapping();
  }

  /**
   * @brief Set the thruster torque coefficients
   * @param[in] p One torque coefficient per thruster
   */
  void set_parameters(const Eigen::Ref<const VectorXs>& p) override {
    if (static_cast<std::size_t>(p.size()) != n_thrusters_) {
      throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                   std::to_string(n_thrusters_) + ")");
    }
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
      thrusters_[i].ctorque = p[i];
    }
    updateMapping();
  }

  /** @brief Return the thruster torque coefficients */
  VectorXs get_parameters() const override {
    VectorXs p(n_thrusters_);
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
      p[i] = thrusters_[i].ctorque;
    }
    return p;
  }

  /** @brief Return the unconstrained torque-coefficient parameterization */
  VectorXs get_parametrization() const override { return get_parameters(); }

  /**
   * @brief Set the physical-parameter Jacobian to identity
   * @param[out] dgamma_dp Torque-coefficient parameter Jacobian
   */
  void updateParametrizationDerivative(
      Eigen::Ref<MatrixXs> dgamma_dp) const override {
    if (static_cast<std::size_t>(dgamma_dp.rows()) != n_thrusters_ ||
        static_cast<std::size_t>(dgamma_dp.cols()) != n_thrusters_) {
      throw_pretty("Invalid argument: dgamma_dp has wrong dimensions");
    }
    dgamma_dp.setIdentity();
  }

  /** @brief Cast the model and its copied thruster descriptors */
  template <typename NewScalar>
  JointDynamicsModelThrusterTpl<NewScalar> cast() const {
    typedef JointDynamicsModelThrusterTpl<NewScalar> ReturnType;
    ReturnType ret(vector_cast<NewScalar>(thrusters_));
    return ret;
  }

  /** @brief Print the thruster joint-dynamics model */
  virtual void print(std::ostream& os) const override {
    os << "JointDynamicsModelThruster {id=" << id_ << ", nq=" << nq_
       << ", nv=" << nv_ << ", nu=" << nu_ << ", nthrusters=" << n_thrusters_
       << "}";
  }

 protected:
  using Base::id_;
  using Base::nq_;
  using Base::nu_;
  using Base::nv_;

 private:
  void updateMapping() {
    W_thrust_.setZero();
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
      const Thruster& thruster = thrusters_[i];
      const Vector3s f_z = thruster.pose.rotation() * Vector3s::UnitZ();
      W_thrust_.template topRows<3>().col(i) = f_z;
      W_thrust_.template middleRows<3>(3).col(i).noalias() =
          thruster.pose.translation().cross(f_z);
      switch (thruster.type) {
        case CW:
          W_thrust_.template middleRows<3>(3).col(i) += thruster.ctorque * f_z;
          break;
        case CCW:
          W_thrust_.template middleRows<3>(3).col(i) -= thruster.ctorque * f_z;
          break;
      }
    }
    Mtau_ = pseudoInverse(W_thrust_);
  }

  std::vector<Thruster> thrusters_;
  std::size_t n_thrusters_;
  MatrixXs W_thrust_;
  MatrixXs Mtau_;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::JointDynamicsModelThrusterTpl)

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_JOINT_THRUSTER_HPP_
