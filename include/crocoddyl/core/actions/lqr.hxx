///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(const std::size_t nx,
                                             const std::size_t nu,
                                             const bool drift_free)
    : Base(boost::make_shared<StateVector>(nx), nu, 0),
      drift_free_(drift_free) {
  // TODO(cmastalli): substitute by random (vectors) and random-orthogonal
  // (matrices)
  A_ = MatrixXs::Identity(nx, nx);
  B_ = MatrixXs::Identity(nx, nu);
  Q_ = MatrixXs::Identity(nx, nx);
  R_ = MatrixXs::Identity(nu, nu);
  N_ = MatrixXs::Identity(nx, nu);
  f_ = VectorXs::Ones(nx);
  q_ = VectorXs::Ones(nx);
  r_ = VectorXs::Ones(nu);
}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::~ActionModelLQRTpl() {}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calc(
    const boost::shared_ptr<ActionDataAbstract>& data,
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
  Data* d = static_cast<Data*>(data.get());

  if (drift_free_) {
    data->xnext.noalias() = A_ * x;
    data->xnext.noalias() += B_ * u;
  } else {
    data->xnext.noalias() = A_ * x;
    data->xnext.noalias() += B_ * u;
    data->xnext += f_;
  }

  // cost = 0.5 * x^T * Q * x + 0.5 * u^T * R * u + x^T * N * u + q^T * x + r^T
  // * u
  d->Q_x_tmp.noalias() = Q_ * x;
  data->cost = Scalar(0.5) * x.dot(d->Q_x_tmp);
  d->R_u_tmp.noalias() = R_ * u;
  data->cost += Scalar(0.5) * u.dot(d->R_u_tmp);
  d->Q_x_tmp.noalias() = N_ * u;
  data->cost += x.dot(d->Q_x_tmp);
  data->cost += q_.dot(x);
  data->cost += r_.dot(u);
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calc(
    const boost::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  d->xnext = x;
  // cost = 0.5 * x^T * Q * x + q^T * x
  d->Q_x_tmp.noalias() = Q_ * x;
  data->cost = Scalar(0.5) * x.dot(d->Q_x_tmp);
  data->cost += q_.dot(x);
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ActionDataAbstract>& data,
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

  data->Fx = A_;
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
void ActionModelLQRTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ActionDataAbstract>& data,
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
boost::shared_ptr<ActionDataAbstractTpl<Scalar>>
ActionModelLQRTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool ActionModelLQRTpl<Scalar>::checkData(
    const boost::shared_ptr<ActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::print(std::ostream& os) const {
  os << "ActionModelLQR {nx=" << state_->get_nx() << ", nu=" << nu_
     << ", drift_free=" << drift_free_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
ActionModelLQRTpl<Scalar>::get_Fx() const {
  return A_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
ActionModelLQRTpl<Scalar>::get_Fu() const {
  return B_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelLQRTpl<Scalar>::get_f0() const {
  return f_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelLQRTpl<Scalar>::get_lx() const {
  return q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelLQRTpl<Scalar>::get_lu() const {
  return r_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
ActionModelLQRTpl<Scalar>::get_Lxx() const {
  return Q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
ActionModelLQRTpl<Scalar>::get_Lxu() const {
  return N_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
ActionModelLQRTpl<Scalar>::get_Luu() const {
  return R_;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Fx(const MatrixXs& Fx) {
  if (static_cast<std::size_t>(Fx.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Fx.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Fx has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  A_ = Fx;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Fu(const MatrixXs& Fu) {
  if (static_cast<std::size_t>(Fu.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Fu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fu has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  B_ = Fu;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_f0(const VectorXs& f0) {
  if (static_cast<std::size_t>(f0.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "f0 has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  f_ = f0;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_lx(const VectorXs& lx) {
  if (static_cast<std::size_t>(lx.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "lx has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  q_ = lx;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_lu(const VectorXs& lu) {
  if (static_cast<std::size_t>(lu.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lu has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  r_ = lu;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Lxx(const MatrixXs& Lxx) {
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
void ActionModelLQRTpl<Scalar>::set_Lxu(const MatrixXs& Lxu) {
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
void ActionModelLQRTpl<Scalar>::set_Luu(const MatrixXs& Luu) {
  if (static_cast<std::size_t>(Luu.rows()) != nu_ ||
      static_cast<std::size_t>(Luu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " +
                        std::to_string(nu_) + "," + std::to_string(nu_) + ")");
  }
  R_ = Luu;
}

}  // namespace crocoddyl
