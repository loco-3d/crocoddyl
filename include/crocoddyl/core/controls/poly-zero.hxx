///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlParametrizationModelPolyZeroTpl<Scalar>::ControlParametrizationModelPolyZeroTpl(const std::size_t nu)
    : Base(nu, nu) {}

template <typename Scalar>
ControlParametrizationModelPolyZeroTpl<Scalar>::~ControlParametrizationModelPolyZeroTpl() {}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::calc(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double,
    const Eigen::Ref<const VectorXs>& p) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                 << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  data->u_diff = p;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::params(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double,
    const Eigen::Ref<const VectorXs>& u) const {
  if (static_cast<std::size_t>(u.size()) != nw_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nw_) + ")");
  }
  data->u_params = u;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::convert_bounds(const Eigen::Ref<const VectorXs>& u_lb,
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
  p_lb = u_lb;
  p_ub = u_ub;
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double,
    const Eigen::Ref<const VectorXs>&) const {
  data->J.setIdentity();
}

template <typename Scalar>
void ControlParametrizationModelPolyZeroTpl<Scalar>::multiplyByJacobian(double, const Eigen::Ref<const VectorXs>&,
                                                                        const Eigen::Ref<const MatrixXs>& A,
                                                                        Eigen::Ref<MatrixXs> out) const {
  if (A.rows() != out.rows() || static_cast<std::size_t>(A.cols()) != nw_ ||
      static_cast<std::size_t>(out.cols()) != np_) {
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
      static_cast<std::size_t>(out.rows()) != np_) {
    throw_pretty("Invalid argument: "
                 << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) +
                        " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + ")");
  }
  out = A;
}

}  // namespace crocoddyl
