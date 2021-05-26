///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlPolyOneTpl<Scalar>::ControlPolyOneTpl(const std::size_t nu) : ControlAbstractTpl<Scalar>(nu, 2*nu) {}

template <typename Scalar>
ControlPolyOneTpl<Scalar>::~ControlPolyOneTpl() {}

template <typename Scalar>
void ControlPolyOneTpl<Scalar>::resize(const std::size_t nu){
  nu_ = nu;
  np_ = 2*nu;
}

template <typename Scalar>
void ControlPolyOneTpl<Scalar>::value(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<VectorXs> u_out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(u_out.size()) != nu_) {
    throw_pretty("Invalid argument: "
                << "u_out has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_out = (1-2*t)*p.head(nu_) + 2*t*p.tail(nu_);
}

template <typename Scalar>
void ControlPolyOneTpl<Scalar>::convert_bounds(const Eigen::Ref<const VectorXs>& u_lb, 
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
  p_lb.head(nu_) = u_lb;
  p_lb.tail(nu_) = u_lb;
  p_ub.head(nu_) = u_ub;
  p_ub.tail(nu_) = u_ub;
}

template <typename Scalar>
void ControlPolyOneTpl<Scalar>::dValue(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<MatrixXs> J_out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(J_out.rows()) != nu_ || static_cast<std::size_t>(J_out.cols()) != np_) {
    throw_pretty("Invalid argument: "
                << "J_out has wrong dimension (it should be " + std::to_string(nu_) + "," + std::to_string(np_) + ")");
  }
  J_out.leftCols(nu_).diagonal()  = MathBase::VectorXs::Constant(nu_, 1-2*t);
  J_out.rightCols(nu_).diagonal() = MathBase::VectorXs::Constant(nu_, 2*t);
}

template <typename Scalar>
void ControlPolyOneTpl<Scalar>::multiplyByDValue(double t, const Eigen::Ref<const VectorXs>& p, 
    const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (A.rows() != out.rows() || A.cols()!=nu_ || out.cols()!=np_) {
    throw_pretty("Invalid argument: "
                << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) 
                + " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + + ")");
  }
  out.leftCols(nu_)  = (1-2*t)*A;
  out.rightCols(nu_) = 2*t*A;
}

template <typename Scalar>
void ControlPolyOneTpl<Scalar>::multiplyDValueTransposeBy(double t, const Eigen::Ref<const VectorXs>& p, 
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
  out.topRows(nu_)    = (1-2*t)*A;
  out.bottomRows(nu_) = 2*t*A;
}

}  // namespace crocoddyl
