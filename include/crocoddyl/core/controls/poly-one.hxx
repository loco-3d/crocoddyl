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
ControlParametrizationModelPolyOneTpl<
    Scalar>::ControlParametrizationModelPolyOneTpl(const std::size_t nw)
    : Base(nw, 2 * nw) {}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::calc(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
    const Scalar t, const Eigen::Ref<const VectorXs>& u) const {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  d->c[1] = Scalar(2.) * t;
  d->c[0] = Scalar(1.) - d->c[1];
  data->w = d->c[0] * u.head(nw_) + d->c[1] * u.tail(nw_);
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::calcDiff(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
    const Scalar, const Eigen::Ref<const VectorXs>&) const {
  Data* d = static_cast<Data*>(data.get());
  data->dw_du.leftCols(nw_).diagonal().array() = d->c[0];
  data->dw_du.rightCols(nw_).diagonal().array() = d->c[1];
}

template <typename Scalar>
std::shared_ptr<ControlParametrizationDataAbstractTpl<Scalar> >
ControlParametrizationModelPolyOneTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::params(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
    const Scalar, const Eigen::Ref<const VectorXs>& w) const {
  if (static_cast<std::size_t>(w.size()) != nw_) {
    throw_pretty(
        "Invalid argument: " << "w has wrong dimension (it should be " +
                                    std::to_string(nw_) + ")");
  }
  data->u.head(nw_) = w;
  data->u.tail(nw_) = w;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::convertBounds(
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
  u_lb.head(nw_) = w_lb;
  u_lb.tail(nw_) = w_lb;
  u_ub.head(nw_) = w_ub;
  u_ub.tail(nw_) = w_ub;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::multiplyByJacobian(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
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
                                             std::to_string(out.cols()) + +")");
  }
  Data* d = static_cast<Data*>(data.get());
  switch (op) {
    case setto:
      out.leftCols(nw_) = d->c[0] * A;
      out.rightCols(nw_) = d->c[1] * A;
      break;
    case addto:
      out.leftCols(nw_) += d->c[0] * A;
      out.rightCols(nw_) += d->c[1] * A;
      break;
    case rmfrom:
      out.leftCols(nw_) -= d->c[0] * A;
      out.rightCols(nw_) -= d->c[1] * A;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
      break;
  }
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::multiplyJacobianTransposeBy(
    const std::shared_ptr<ControlParametrizationDataAbstract>& data,
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
  Data* d = static_cast<Data*>(data.get());
  switch (op) {
    case setto:
      out.topRows(nw_) = d->c[0] * A;
      out.bottomRows(nw_) = d->c[1] * A;
      break;
    case addto:
      out.topRows(nw_) += d->c[0] * A;
      out.bottomRows(nw_) += d->c[1] * A;
      break;
    case rmfrom:
      out.topRows(nw_) -= d->c[0] * A;
      out.bottomRows(nw_) -= d->c[1] * A;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
      break;
  }
}

template <typename Scalar>
template <typename NewScalar>
ControlParametrizationModelPolyOneTpl<NewScalar>
ControlParametrizationModelPolyOneTpl<Scalar>::cast() const {
  typedef ControlParametrizationModelPolyOneTpl<NewScalar> ReturnType;
  ReturnType ret(nw_);
  return ret;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::print(
    std::ostream& os) const {
  os << "ControlParametrizationModelPolyZero {nw=" << nw_ << "}";
}

}  // namespace crocoddyl
