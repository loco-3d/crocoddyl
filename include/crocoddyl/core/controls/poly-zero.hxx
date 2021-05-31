///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlPolyZeroTpl<Scalar>::ControlPolyZeroTpl(const std::size_t nu) : ControlAbstractTpl<Scalar>(nu, nu) {}

template <typename Scalar>
ControlPolyZeroTpl<Scalar>::~ControlPolyZeroTpl() {}

template <typename Scalar>
void ControlPolyZeroTpl<Scalar>::resize(const std::size_t nu){
  nu_ = nu;
  np_ = nu;
}

template <typename Scalar>
void ControlPolyZeroTpl<Scalar>::value(double, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<VectorXs> u_out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(u_out.size()) != nu_) {
    throw_pretty("Invalid argument: "
                << "u_out has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_out = p;
}

template <typename Scalar>
void ControlPolyZeroTpl<Scalar>::value_inv(double t, const Eigen::Ref<const VectorXs>& u, Eigen::Ref<VectorXs> p_out) const{
  if (static_cast<std::size_t>(p_out.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p_out has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  p_out = u;
}

template <typename Scalar>
void ControlPolyZeroTpl<Scalar>::convert_bounds(const Eigen::Ref<const VectorXs>& u_lb, 
    const Eigen::Ref<const VectorXs>& u_ub, Eigen::Ref<VectorXs> p_lb, Eigen::Ref<VectorXs> p_ub) const{
  if (static_cast<std::size_t>(p_lb.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p_lb has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(p_ub.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p_ub has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(u_lb.size()) != nu_) {
    throw_pretty("Invalid argument: "
                << "u_lb has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(u_ub.size()) != nu_) {
    throw_pretty("Invalid argument: "
                << "u_ub has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  p_lb = u_lb;
  p_ub = u_ub;
}

template <typename Scalar>
void ControlPolyZeroTpl<Scalar>::dValue(double, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<MatrixXs> J_out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(J_out.rows()) != nu_ || static_cast<std::size_t>(J_out.cols()) != np_) {
    throw_pretty("Invalid argument: "
                << "J_out has wrong dimension (it should be " + std::to_string(nu_) + "," + std::to_string(np_) + ")");
  }
  J_out.setIdentity();
}

template <typename Scalar>
void ControlPolyZeroTpl<Scalar>::multiplyByDValue(double, const Eigen::Ref<const VectorXs>& p, 
    const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (A.rows() != out.rows() || static_cast<std::size_t>(A.cols())!=nu_ || static_cast<std::size_t>(out.cols())!=np_) {
    throw_pretty("Invalid argument: "
                << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) 
                + " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + ")");
  }
  out = A;
}

template <typename Scalar>
void ControlPolyZeroTpl<Scalar>::multiplyDValueTransposeBy(double, const Eigen::Ref<const VectorXs>& p, 
    const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (A.cols() != out.cols() || static_cast<std::size_t>(A.rows())!=nu_ || static_cast<std::size_t>(out.rows())!=np_) {
    throw_pretty("Invalid argument: "
                << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) 
                + " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + ")");
  }
  out = A;
}

}  // namespace crocoddyl
