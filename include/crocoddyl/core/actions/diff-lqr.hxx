///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>::DifferentialActionModelLQRTpl(const std::size_t nq, const std::size_t nu,
                                                                     const bool drift_free)
    : Base(boost::make_shared<StateVector>(2 * nq), nu), drift_free_(drift_free) {
  // TODO(cmastalli): substitute by random (vectors) and random-orthogonal (matrices)
  Fq_ = MatrixXs::Identity(state_->get_nq(), state_->get_nq());
  Fv_ = MatrixXs::Identity(state_->get_nv(), state_->get_nv());
  Fu_ = MatrixXs::Identity(state_->get_nq(), nu_);
  f0_ = VectorXs::Ones(state_->get_nv());
  Lxx_ = MatrixXs::Identity(state_->get_nx(), state_->get_nx());
  Lxu_ = MatrixXs::Identity(state_->get_nx(), nu_);
  Luu_ = MatrixXs::Identity(nu_, nu_);
  lx_ = VectorXs::Ones(state_->get_nx());
  lu_ = VectorXs::Ones(nu_);
}

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>::~DifferentialActionModelLQRTpl() {}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x,
                                                 const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());

  if (drift_free_) {
    data->xout = Fq_ * q + Fv_ * v + Fu_ * u;
  } else {
    data->xout = Fq_ * q + Fv_ * v + Fu_ * u + f0_;
  }
  data->cost =
      Scalar(0.5) * x.dot(Lxx_ * x) + Scalar(0.5) * u.dot(Luu_ * u) + x.dot(Lxu_ * u) + lx_.dot(x) + lu_.dot(u);
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>& x,
                                                     const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  data->Lx = lx_ + Lxx_ * x + Lxu_ * u;
  data->Lu = lu_ + Lxu_.transpose() * x + Luu_ * u;
  data->Fx.leftCols(state_->get_nq()) = Fq_;
  data->Fx.rightCols(state_->get_nv()) = Fv_;
  data->Fu = Fu_;
  data->Lxx = Lxx_;
  data->Lxu = Lxu_;
  data->Luu = Luu_;
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> > DifferentialActionModelLQRTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelLQRTpl<Scalar>::checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::print(std::ostream& os) const {
  os << "DifferentialActionModelLQR {nq=" << state_->get_nq() << ", nu=" << nu_ << ", drift_free=" << drift_free_
     << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& DifferentialActionModelLQRTpl<Scalar>::get_Fq() const {
  return Fq_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& DifferentialActionModelLQRTpl<Scalar>::get_Fv() const {
  return Fv_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& DifferentialActionModelLQRTpl<Scalar>::get_Fu() const {
  return Fu_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelLQRTpl<Scalar>::get_f0() const {
  return f0_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelLQRTpl<Scalar>::get_lx() const {
  return lx_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelLQRTpl<Scalar>::get_lu() const {
  return lu_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& DifferentialActionModelLQRTpl<Scalar>::get_Lxx() const {
  return Lxx_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& DifferentialActionModelLQRTpl<Scalar>::get_Lxu() const {
  return Lxu_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& DifferentialActionModelLQRTpl<Scalar>::get_Luu() const {
  return Luu_;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Fq(const MatrixXs& Fq) {
  if (static_cast<std::size_t>(Fq.rows()) != state_->get_nq() ||
      static_cast<std::size_t>(Fq.cols()) != state_->get_nq()) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " + std::to_string(state_->get_nq()) + "," +
                        std::to_string(state_->get_nq()) + ")");
  }
  Fq_ = Fq;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Fv(const MatrixXs& Fv) {
  if (static_cast<std::size_t>(Fv.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(Fv.cols()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "Fv has wrong dimension (it should be " + std::to_string(state_->get_nv()) + "," +
                        std::to_string(state_->get_nv()) + ")");
  }
  Fv_ = Fv;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Fu(const MatrixXs& Fu) {
  if (static_cast<std::size_t>(Fu.rows()) != state_->get_nq() || static_cast<std::size_t>(Fu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fu has wrong dimension (it should be " + std::to_string(state_->get_nq()) + "," +
                        std::to_string(nu_) + ")");
  }
  Fu_ = Fu;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_f0(const VectorXs& f0) {
  if (static_cast<std::size_t>(f0.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "f0 has wrong dimension (it should be " + std::to_string(state_->get_nv()) + ")");
  }
  f0_ = f0;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_lx(const VectorXs& lx) {
  if (static_cast<std::size_t>(lx.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "lx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  lx_ = lx;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_lu(const VectorXs& lu) {
  if (static_cast<std::size_t>(lu.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lu has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  lu_ = lu;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Lxx(const MatrixXs& Lxx) {
  if (static_cast<std::size_t>(Lxx.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Lxx.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Lxx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  Lxx_ = Lxx;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Lxu(const MatrixXs& Lxu) {
  if (static_cast<std::size_t>(Lxu.rows()) != state_->get_nx() || static_cast<std::size_t>(Lxu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Lxu has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  Lxu_ = Lxu;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Luu(const MatrixXs& Luu) {
  if (static_cast<std::size_t>(Luu.rows()) != nu_ || static_cast<std::size_t>(Luu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " + std::to_string(nu_) + "," + std::to_string(nu_) + ")");
  }
  Luu_ = Luu;
}

}  // namespace crocoddyl
