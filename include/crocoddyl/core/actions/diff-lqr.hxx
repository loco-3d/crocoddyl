///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>::DifferentialActionModelLQRTpl(
    const std::size_t nq, const std::size_t nu, const bool drift_free)
    : Base(boost::make_shared<StateVector>(2 * nq), nu),
      drift_free_(drift_free) {
  // TODO(cmastalli): substitute by random (vectors) and random-orthogonal
  // (matrices)
  Aq_ = MatrixXs::Identity(state_->get_nq(), state_->get_nq());
  Av_ = MatrixXs::Identity(state_->get_nv(), state_->get_nv());
  B_ = MatrixXs::Identity(state_->get_nq(), nu_);
  Q_ = MatrixXs::Identity(state_->get_nx(), state_->get_nx());
  R_ = MatrixXs::Identity(nu_, nu_);
  N_ = MatrixXs::Identity(state_->get_nx(), nu_);
  f_ = VectorXs::Ones(state_->get_nv());
  q_ = VectorXs::Ones(state_->get_nx());
  r_ = VectorXs::Ones(nu_);
}

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>::~DifferentialActionModelLQRTpl() {}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  if (drift_free_) {
    data->xout.noalias() = Aq_ * q;
    data->xout.noalias() += Av_ * v;
    data->xout.noalias() += B_ * u;
  } else {
    data->xout.noalias() = Aq_ * q;
    data->xout.noalias() += Av_ * v;
    data->xout.noalias() += B_ * u;
    data->xout += f_;
  }
  // cost = 0.5 * x^T * Q * x + 0.5 * u^T * R * u + x^T * N * u + q^T * x + r^T
  // * u
  data->cost = Scalar(0.5) * x.dot(Q_ * x);
  data->cost += Scalar(0.5) * u.dot(R_ * u);
  data->cost += x.dot(N_ * u);
  data->cost += q_.dot(x);
  data->cost += r_.dot(u);
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }

  // cost = 0.5 * x^T * Q * x + q^T * x
  data->cost = Scalar(0.5) * x.dot(Q_ * x);
  data->cost += q_.dot(x);
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }

  data->Fx.leftCols(state_->get_nq()) = Aq_;
  data->Fx.rightCols(state_->get_nv()) = Av_;
  data->Fu = B_;
  data->Lxx = Q_;
  data->Luu = R_;
  data->Lxu = N_;
  data->Lx = q_;
  data->Lx.noalias() += Q_ * x;
  data->Lx.noalias() += N_ * u;
  data->Lu = r_;
  data->Lu.noalias() += N_.transpose() * x;
  data->Lu.noalias() += R_ * u;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }

  data->Lxx = Q_;
  data->Lx = q_;
  data->Lx.noalias() += Q_ * x;
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelLQRTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelLQRTpl<Scalar>::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::print(std::ostream& os) const {
  os << "DifferentialActionModelLQR {nq=" << state_->get_nq() << ", nu=" << nu_
     << ", drift_free=" << drift_free_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Fq() const {
  return Aq_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Fv() const {
  return Av_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Fu() const {
  return B_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_f0() const {
  return f_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_lx() const {
  return q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_lu() const {
  return r_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Lxx() const {
  return Q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Lxu() const {
  return N_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Luu() const {
  return R_;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Fq(const MatrixXs& Fq) {
  if (static_cast<std::size_t>(Fq.rows()) != state_->get_nq() ||
      static_cast<std::size_t>(Fq.cols()) != state_->get_nq()) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " +
                        std::to_string(state_->get_nq()) + "," +
                        std::to_string(state_->get_nq()) + ")");
  }
  Aq_ = Fq;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Fv(const MatrixXs& Fv) {
  if (static_cast<std::size_t>(Fv.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(Fv.cols()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "Fv has wrong dimension (it should be " +
                        std::to_string(state_->get_nv()) + "," +
                        std::to_string(state_->get_nv()) + ")");
  }
  Av_ = Fv;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Fu(const MatrixXs& Fu) {
  if (static_cast<std::size_t>(Fu.rows()) != state_->get_nq() ||
      static_cast<std::size_t>(Fu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fu has wrong dimension (it should be " +
                        std::to_string(state_->get_nq()) + "," +
                        std::to_string(nu_) + ")");
  }
  B_ = Fu;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_f0(const VectorXs& f0) {
  if (static_cast<std::size_t>(f0.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "f0 has wrong dimension (it should be " +
                        std::to_string(state_->get_nv()) + ")");
  }
  f_ = f0;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_lx(const VectorXs& lx) {
  if (static_cast<std::size_t>(lx.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "lx has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  q_ = lx;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_lu(const VectorXs& lu) {
  if (static_cast<std::size_t>(lu.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lu has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  r_ = lu;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Lxx(const MatrixXs& Lxx) {
  if (static_cast<std::size_t>(Lxx.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Lxx.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Lxx has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  Q_ = Lxx;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Lxu(const MatrixXs& Lxu) {
  if (static_cast<std::size_t>(Lxu.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Lxu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Lxu has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  N_ = Lxu;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Luu(const MatrixXs& Luu) {
  if (static_cast<std::size_t>(Luu.rows()) != nu_ ||
      static_cast<std::size_t>(Luu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " +
                        std::to_string(nu_) + "," + std::to_string(nu_) + ")");
  }
  R_ = Luu;
}

}  // namespace crocoddyl
