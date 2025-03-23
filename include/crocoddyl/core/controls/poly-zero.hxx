///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, University of Trento,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelPolyZeroTpl<
    Scalar>::ControlParametrizationModelPolyZeroTpl(const std::size_t nw)
    : Base(nw, nw) {}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::calc(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
    const Scalar, const Eigen::Ref<const VectorXs>& u) const {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  data->w = u;
}

template <typename Scalar>
#ifndef NDEBUG
void ControlParametrizationModelPolyZeroTpl<Scalar>::calcDiff(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
    const Scalar, const Eigen::Ref<const VectorXs>&) const {
#else
void ControlParametrizationModelPolyZeroTpl<Scalar>::calcDiff(
    const std::shared_ptr<ControlParametrizationDataAbstract>&, const Scalar,
    const Eigen::Ref<const VectorXs>&) const {
#endif
  // The Hessian has constant values which were set in createData.
  assert_pretty(MatrixXs(data->dw_du).isApprox(MatrixXs::Identity(nu_, nu_)),
                "dw_du has wrong value");
}

template <typename Scalar>
std::shared_ptr<ControlParametrizationDataAbstractTpl<Scalar> >
ControlParametrizationModelPolyZeroTpl<Scalar>::createData() {
  std::shared_ptr<Data> data =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  data->dw_du.setIdentity();
  return data;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::params(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
    const Scalar, const Eigen::Ref<const VectorXs>& w) const {
  if (static_cast<std::size_t>(w.size()) != nw_) {
    throw_pretty(
        "Invalid argument: " << "w has wrong dimension (it should be " +
                                    std::to_string(nw_) + ")");
  }
  data->u = w;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::convertBounds(
    const Eigen::Ref<const VectorXs>& w_lb,
    const Eigen::Ref<const VectorXs>& w_ub, Eigen::Ref<VectorXs> u_lb,
    Eigen::Ref<VectorXs> u_ub) const {
  if (static_cast<std::size_t>(u_lb.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u_lb has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(u_ub.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u_ub has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(w_lb.size()) != nw_) {
    throw_pretty(
        "Invalid argument: " << "w_lb has wrong dimension (it should be " +
                                    std::to_string(nw_) + ")");
  }
  if (static_cast<std::size_t>(w_ub.size()) != nw_) {
    throw_pretty(
        "Invalid argument: " << "w_ub has wrong dimension (it should be " +
                                    std::to_string(nw_) + ")");
  }
  u_lb = w_lb;
  u_ub = w_ub;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::multiplyByJacobian(
    const std::shared_ptr<ControlParametrizationDataAbstract>&,
    const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
    const AssignmentOp op) const {
  assert_pretty(is_a_AssignmentOp(op),
                ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (A.rows() != out.rows() || static_cast<std::size_t>(A.cols()) != nw_ ||
      static_cast<std::size_t>(out.cols()) != nu_) {
    throw_pretty("Invalid argument: " << "A and out have wrong dimensions (" +
                                             std::to_string(A.rows()) + "," +
                                             std::to_string(A.cols()) +
                                             " and " +
                                             std::to_string(out.rows()) + "," +
                                             std::to_string(out.cols()) + ")");
  }
  switch (op) {
    case setto:
      out = A;
      break;
    case addto:
      out += A;
      break;
    case rmfrom:
      out -= A;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
      break;
  }
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::
    multiplyJacobianTransposeBy(
        const std::shared_ptr<ControlParametrizationDataAbstract>&,
        const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
        const AssignmentOp op) const {
  assert_pretty(is_a_AssignmentOp(op),
                ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (A.cols() != out.cols() || static_cast<std::size_t>(A.rows()) != nw_ ||
      static_cast<std::size_t>(out.rows()) != nu_) {
    throw_pretty("Invalid argument: " << "A and out have wrong dimensions (" +
                                             std::to_string(A.rows()) + "," +
                                             std::to_string(A.cols()) +
                                             " and " +
                                             std::to_string(out.rows()) + "," +
                                             std::to_string(out.cols()) + ")");
  }
  switch (op) {
    case setto:
      out = A;
      break;
    case addto:
      out += A;
      break;
    case rmfrom:
      out -= A;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
      break;
  }
}

template <typename Scalar>
template <typename NewScalar>
ControlParametrizationModelPolyZeroTpl<NewScalar>
ControlParametrizationModelPolyZeroTpl<Scalar>::cast() const {
  typedef ControlParametrizationModelPolyZeroTpl<NewScalar> ReturnType;
  ReturnType ret(nw_);
  return ret;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::print(
    std::ostream& os) const {
  os << "ControlParametrizationModelPolyZero {nw=" << nw_ << "}";
}

}  // namespace crocoddyl
