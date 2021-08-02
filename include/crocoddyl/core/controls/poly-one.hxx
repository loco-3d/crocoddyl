///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelPolyOneTpl<Scalar>::ControlParametrizationModelPolyOneTpl(const std::size_t nw)
    : Base(nw, 2 * nw) {}

template <typename Scalar>
ControlParametrizationModelPolyOneTpl<Scalar>::~ControlParametrizationModelPolyOneTpl() {}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::calc(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
    const Eigen::Ref<const VectorXs>& u) const {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  data->w = (1 - 2 * t) * u.head(nw_) + 2 * t * u.tail(nw_);
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
    const Eigen::Ref<const VectorXs>&) const {
  data->dw_du.leftCols(nw_).diagonal() = MathBase::VectorXs::Constant(nw_, 1 - 2 * t);
  data->dw_du.rightCols(nw_).diagonal() = MathBase::VectorXs::Constant(nw_, 2 * t);
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::params(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double,
    const Eigen::Ref<const VectorXs>& w) const {
  if (static_cast<std::size_t>(w.size()) != nw_) {
    throw_pretty("Invalid argument: "
                 << "w has wrong dimension (it should be " + std::to_string(nw_) + ")");
  }
  data->u.head(nw_) = w;
  data->u.tail(nw_) = w;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::convertBounds(const Eigen::Ref<const VectorXs>& w_lb,
                                                                  const Eigen::Ref<const VectorXs>& w_ub,
                                                                  Eigen::Ref<VectorXs> u_lb,
                                                                  Eigen::Ref<VectorXs> u_ub) const {
  if (static_cast<std::size_t>(u_lb.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u_lb has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(u_ub.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u_ub has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(w_lb.size()) != nw_) {
    throw_pretty("Invalid argument: "
                 << "w_lb has wrong dimension (it should be " + std::to_string(nw_) + ")");
  }
  if (static_cast<std::size_t>(w_ub.size()) != nw_) {
    throw_pretty("Invalid argument: "
                 << "w_ub has wrong dimension (it should be " + std::to_string(nw_) + ")");
  }
  u_lb.head(nw_) = w_lb;
  u_lb.tail(nw_) = w_lb;
  u_ub.head(nw_) = w_ub;
  u_ub.tail(nw_) = w_ub;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>&,
                                                                       const Eigen::Ref<const MatrixXs>& A,
                                                                       Eigen::Ref<MatrixXs> out) const {
  if (A.rows() != out.rows() || static_cast<std::size_t>(A.cols()) != nw_ ||
      static_cast<std::size_t>(out.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) +
                        " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + +")");
  }
  out.leftCols(nw_) = (1 - 2 * t) * A;
  out.rightCols(nw_) = 2 * t * A;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::multiplyJacobianTransposeBy(double t,
                                                                                const Eigen::Ref<const VectorXs>&,
                                                                                const Eigen::Ref<const MatrixXs>& A,
                                                                                Eigen::Ref<MatrixXs> out) const {
  if (A.cols() != out.cols() || static_cast<std::size_t>(A.rows()) != nw_ ||
      static_cast<std::size_t>(out.rows()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) +
                        " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + ")");
  }
  out.topRows(nw_) = (1 - 2 * t) * A;
  out.bottomRows(nw_) = 2 * t * A;
}

}  // namespace crocoddyl
