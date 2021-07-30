///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelPolyOneTpl<Scalar>::ControlParametrizationModelPolyOneTpl(const std::size_t nu)
    : Base(nu, 2 * nu) {}

template <typename Scalar>
ControlParametrizationModelPolyOneTpl<Scalar>::~ControlParametrizationModelPolyOneTpl() {}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::calc(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
    const Eigen::Ref<const VectorXs>& p) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                 << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  data->u_diff = (1 - 2 * t) * p.head(nw_) + 2 * t * p.tail(nw_);
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::params(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double,
    const Eigen::Ref<const VectorXs>& u) const {
  if (static_cast<std::size_t>(u.size()) != nw_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nw_) + ")");
  }
  data->u_params.head(nw_) = u;
  data->u_params.tail(nw_) = u;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::convert_bounds(const Eigen::Ref<const VectorXs>& u_lb,
                                                                   const Eigen::Ref<const VectorXs>& u_ub,
                                                                   Eigen::Ref<VectorXs> p_lb,
                                                                   Eigen::Ref<VectorXs> p_ub) const {
  if (static_cast<std::size_t>(p_lb.size()) != np_) {
    throw_pretty("Invalid argument: "
                 << "p_lb has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(p_ub.size()) != np_) {
    throw_pretty("Invalid argument: "
                 << "p_ub has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(u_lb.size()) != nw_) {
    throw_pretty("Invalid argument: "
                 << "u_lb has wrong dimension (it should be " + std::to_string(nw_) + ")");
  }
  if (static_cast<std::size_t>(u_ub.size()) != nw_) {
    throw_pretty("Invalid argument: "
                 << "u_ub has wrong dimension (it should be " + std::to_string(nw_) + ")");
  }
  p_lb.head(nw_) = u_lb;
  p_lb.tail(nw_) = u_lb;
  p_ub.head(nw_) = u_ub;
  p_ub.tail(nw_) = u_ub;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
    const Eigen::Ref<const VectorXs>& p) const {
  data->J.leftCols(nw_).diagonal() = MathBase::VectorXs::Constant(nw_, 1 - 2 * t);
  data->J.rightCols(nw_).diagonal() = MathBase::VectorXs::Constant(nw_, 2 * t);
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>& p,
                                                                       const Eigen::Ref<const MatrixXs>& A,
                                                                       Eigen::Ref<MatrixXs> out) const {
  if (A.rows() != out.rows() || static_cast<std::size_t>(A.cols()) != nw_ ||
      static_cast<std::size_t>(out.cols()) != np_) {
    throw_pretty("Invalid argument: "
                 << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) +
                        " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + +")");
  }
  out.leftCols(nw_) = (1 - 2 * t) * A;
  out.rightCols(nw_) = 2 * t * A;
}

template <typename Scalar>
void ControlParametrizationModelPolyOneTpl<Scalar>::multiplyJacobianTransposeBy(double t,
                                                                                const Eigen::Ref<const VectorXs>& p,
                                                                                const Eigen::Ref<const MatrixXs>& A,
                                                                                Eigen::Ref<MatrixXs> out) const {
  if (A.cols() != out.cols() || static_cast<std::size_t>(A.rows()) != nw_ ||
      static_cast<std::size_t>(out.rows()) != np_) {
    throw_pretty("Invalid argument: "
                 << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) +
                        " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + ")");
  }
  out.topRows(nw_) = (1 - 2 * t) * A;
  out.bottomRows(nw_) = 2 * t * A;
}

}  // namespace crocoddyl
