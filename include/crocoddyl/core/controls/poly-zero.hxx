///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelPolyZeroTpl<Scalar>::ControlParametrizationModelPolyZeroTpl(const std::size_t nw)
    : Base(nw, nw) {}

template <typename Scalar>
ControlParametrizationModelPolyZeroTpl<Scalar>::~ControlParametrizationModelPolyZeroTpl() {}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::calc(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double,
    const Eigen::Ref<const VectorXs>& u) const {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  data->w = u;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double,
    const Eigen::Ref<const VectorXs>&) const {
  data->dw_du.setIdentity();
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::params(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double,
    const Eigen::Ref<const VectorXs>& w) const {
  if (static_cast<std::size_t>(w.size()) != nw_) {
    throw_pretty("Invalid argument: "
                 << "w has wrong dimension (it should be " + std::to_string(nw_) + ")");
  }
  data->u = w;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::convertBounds(const Eigen::Ref<const VectorXs>& w_lb,
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
  u_lb = w_lb;
  u_ub = w_ub;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::multiplyByJacobian(double, const Eigen::Ref<const VectorXs>&,
                                                                        const Eigen::Ref<const MatrixXs>& A,
                                                                        Eigen::Ref<MatrixXs> out) const {
  if (A.rows() != out.rows() || static_cast<std::size_t>(A.cols()) != nw_ ||
      static_cast<std::size_t>(out.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) +
                        " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + ")");
  }
  out = A;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::multiplyJacobianTransposeBy(double,
                                                                                 const Eigen::Ref<const VectorXs>&,
                                                                                 const Eigen::Ref<const MatrixXs>& A,
                                                                                 Eigen::Ref<MatrixXs> out) const {
  if (A.cols() != out.cols() || static_cast<std::size_t>(A.rows()) != nw_ ||
      static_cast<std::size_t>(out.rows()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) +
                        " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + ")");
  }
  out = A;
}

}  // namespace crocoddyl
