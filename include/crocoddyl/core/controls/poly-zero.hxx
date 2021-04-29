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
  if (A.rows() != out.rows() || A.cols()!=nu_ || out.cols()!=np_) {
    throw_pretty("Invalid argument: "
                << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) 
                + " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + ")");
  }
  out = A;
}

}  // namespace crocoddyl
