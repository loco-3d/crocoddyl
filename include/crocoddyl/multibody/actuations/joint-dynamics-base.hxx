///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
JointDynamicsModelAbstractTpl<Scalar>::JointDynamicsModelAbstractTpl(
    const pinocchio::JointIndex id, const std::size_t nq, const std::size_t nv,
    const std::size_t nu)
    : id_(id), nq_(nq), nv_(nv), nu_(nu) {}

template <typename Scalar>
std::shared_ptr<JointDynamicsDataAbstractTpl<Scalar> >
JointDynamicsModelAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<JointDynamicsDataAbstract>(
      Eigen::aligned_allocator<JointDynamicsDataAbstract>(), this);
}

template <typename Scalar>
std::size_t JointDynamicsModelAbstractTpl<Scalar>::get_np() const {
  return 0;
}

template <typename Scalar>
void JointDynamicsModelAbstractTpl<Scalar>::set_parameters(
    const Eigen::Ref<const VectorXs>& p) {
  if (p.size() != 0) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be 0)");
  }
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
JointDynamicsModelAbstractTpl<Scalar>::get_parameters() const {
  return VectorXs::Zero(0);
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
JointDynamicsModelAbstractTpl<Scalar>::get_parametrization() const {
  return VectorXs::Zero(0);
}

template <typename Scalar>
void JointDynamicsModelAbstractTpl<Scalar>::updateParametrizationDerivative(
    Eigen::Ref<MatrixXs> dgamma_dp) const {
  if (dgamma_dp.rows() != 0 || dgamma_dp.cols() != 0) {
    throw_pretty(
        "Invalid argument: dgamma_dp has wrong dimensions (it should be 0x0)");
  }
}

template <typename Scalar>
void JointDynamicsModelAbstractTpl<Scalar>::computeJointTorqueRegressor(
    Eigen::Ref<MatrixXs> joint_dtau_dp, const Eigen::Ref<const VectorXs>& q,
    const Eigen::Ref<const VectorXs>& v,
    const Eigen::Ref<const VectorXs>& u) const {
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
      static_cast<std::size_t>(joint_dtau_dp.cols()) != 0) {
    throw_pretty(
        "Invalid argument: joint_dtau_dp has wrong dimensions (it should be " +
        std::to_string(nv_) + "x0)");
  }
  joint_dtau_dp.setZero();
}

template <typename Scalar>
pinocchio::JointIndex JointDynamicsModelAbstractTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
std::size_t JointDynamicsModelAbstractTpl<Scalar>::get_nq() const {
  return nq_;
}

template <typename Scalar>
std::size_t JointDynamicsModelAbstractTpl<Scalar>::get_nv() const {
  return nv_;
}

template <typename Scalar>
std::size_t JointDynamicsModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const JointDynamicsModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

template <typename Scalar>
void JointDynamicsModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

}  // namespace crocoddyl
