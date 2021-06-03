///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ControlPolyTwoRK4Tpl<Scalar>::ControlPolyTwoRK4Tpl(const std::size_t nu) : ControlAbstractTpl<Scalar>(nu, 3*nu) {}

template <typename Scalar>
ControlPolyTwoRK4Tpl<Scalar>::~ControlPolyTwoRK4Tpl() {}

template <typename Scalar>
void ControlPolyTwoRK4Tpl<Scalar>::resize(const std::size_t nu){
  nu_ = nu;
  np_ = 3*nu;
}

template <typename Scalar>
void ControlPolyTwoRK4Tpl<Scalar>::value(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<VectorXs> u_out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(u_out.size()) != nu_) {
    throw_pretty("Invalid argument: "
                << "u_out has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs> > &p0 = p.head(nu_);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs> > &p1 = p.segment(nu_, nu_);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs> > &p2 = p.tail(nu_);
  // u_out = (t*t)*(2*p2-4*p1+2*p0) + t*(4*p1-p2-3*p0) + p0;
  Scalar t2 = t*t;
  u_out = (2*t2-t)*p2 + (4*(t-t2))*p1 + (1-3*t+2*t2)*p0;
}

template <typename Scalar>
void ControlPolyTwoRK4Tpl<Scalar>::value_inv(double, const Eigen::Ref<const VectorXs>& u, Eigen::Ref<VectorXs> p_out) const{
  if (static_cast<std::size_t>(p_out.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p_out has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  p_out.head(nu_)         = u;
  p_out.segment(nu_, nu_) = u;
  p_out.tail(nu_)         = u;
}

template <typename Scalar>
void ControlPolyTwoRK4Tpl<Scalar>::convert_bounds(const Eigen::Ref<const VectorXs>& u_lb, 
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
  p_lb.head(nu_)          = u_lb;
  p_lb.segment(nu_, nu_)  = u_lb;
  p_lb.tail(nu_)          = u_lb;
  p_ub.head(nu_)          = u_ub;
  p_ub.segment(nu_, nu_)  = u_ub;
  p_ub.tail(nu_)          = u_ub;
}

template <typename Scalar>
void ControlPolyTwoRK4Tpl<Scalar>::dValue(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<MatrixXs> J_out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(J_out.rows()) != nu_ || static_cast<std::size_t>(J_out.cols()) != np_) {
    throw_pretty("Invalid argument: "
                << "J_out has wrong dimension (it should be " + std::to_string(nu_) + "," + std::to_string(np_) + ")");
  }
  // u_out = (2*t2-t)*p2 + (4*(t-t2))*p1 + (1-3*t+2*t2)*p0;
  Scalar t2 = t*t;
  J_out.leftCols(nu_).diagonal()        = MathBase::VectorXs::Constant(nu_, 1-3*t+2*t2);
  J_out.middleCols(nu_,nu_).diagonal()  = MathBase::VectorXs::Constant(nu_, 4*(t-t2));
  J_out.rightCols(nu_).diagonal()       = MathBase::VectorXs::Constant(nu_, 2*t2-t);
}

template <typename Scalar>
void ControlPolyTwoRK4Tpl<Scalar>::multiplyByDValue(double t, const Eigen::Ref<const VectorXs>& p, 
    const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: "
                << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
  }
  if (A.rows() != out.rows() || static_cast<std::size_t>(A.cols())!=nu_ || static_cast<std::size_t>(out.cols())!=np_) {
    throw_pretty("Invalid argument: "
                << "A and out have wrong dimensions (" + std::to_string(A.rows()) + "," + std::to_string(A.cols()) 
                + " and " + std::to_string(out.rows()) + "," + std::to_string(out.cols()) + + ")");
  }
  Scalar t2 = t*t;
  out.leftCols(nu_)       = (1-3*t+2*t2)*A;
  out.middleCols(nu_,nu_) = (4*(t-t2))*A;
  out.rightCols(nu_)      = (2*t2-t)*A;
}

template <typename Scalar>
void ControlPolyTwoRK4Tpl<Scalar>::multiplyDValueTransposeBy(double t, const Eigen::Ref<const VectorXs>& p, 
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
  Scalar t2 = t*t;
  out.topRows(nu_)        = (1-3*t+2*t2)*A;
  out.middleRows(nu_,nu_) = (4*(t-t2))*A;
  out.bottomRows(nu_)     = (2*t2-t)*A;
}

}  // namespace crocoddyl
