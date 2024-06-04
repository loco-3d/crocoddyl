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
      Aq_(MatrixXs::Identity(nq, nq)),
      Av_(MatrixXs::Identity(nq, nq)),
      B_(MatrixXs::Identity(nq, nu)),
      Q_(MatrixXs::Identity(2 * nq, 2 * nq)),
      R_(MatrixXs::Identity(nu, nu)),
      N_(MatrixXs::Zero(2 * nq, nu)),
      f_(drift_free ? VectorXs::Zero(nq) : VectorXs::Ones(nq)),
      q_(VectorXs::Ones(2 * nq)),
      r_(VectorXs::Ones(nu)),
      drift_free_(drift_free) {
  // TODO(cmastalli): substitute by random (vectors) and random-orthogonal
  // (matrices)
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

  data->xout.noalias() = Aq_ * q;
  data->xout.noalias() += Av_ * v;
  data->xout.noalias() += B_ * u;
  data->xout += f_;

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
DifferentialActionModelLQRTpl<Scalar>::get_Aq() const {
  return Aq_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Av() const {
  return Av_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_B() const {
  return B_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_f() const {
  return f_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Q() const {
  return Q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_R() const {
  return R_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_N() const {
  return N_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_q() const {
  return q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_r() const {
  return r_;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Aq(const MatrixXs& Aq) {
  if (static_cast<std::size_t>(Aq.rows()) != state_->get_nq() ||
      static_cast<std::size_t>(Aq.cols()) != state_->get_nq()) {
    throw_pretty("Invalid argument: "
                 << "Aq has wrong dimension (it should be " +
                        std::to_string(state_->get_nq()) + "," +
                        std::to_string(state_->get_nq()) + ")");
  }
  Aq_ = Aq;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Av(const MatrixXs& Av) {
  if (static_cast<std::size_t>(Av.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(Av.cols()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "Av has wrong dimension (it should be " +
                        std::to_string(state_->get_nv()) + "," +
                        std::to_string(state_->get_nv()) + ")");
  }
  Av_ = Av;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_B(const MatrixXs& B) {
  if (static_cast<std::size_t>(B.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(B.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "B has wrong dimension (it should be " +
                        std::to_string(state_->get_nv()) + "," +
                        std::to_string(nu_) + ")");
  }
  B_ = B;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_f(const VectorXs& f) {
  if (static_cast<std::size_t>(f.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "f has wrong dimension (it should be " +
                        std::to_string(state_->get_nv()) + ")");
  }
  f_ = f;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_Q(const MatrixXs& Q) {
  if (static_cast<std::size_t>(Q.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Q.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Q has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  Q_ = Q;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_R(const MatrixXs& R) {
  if (static_cast<std::size_t>(R.rows()) != nu_ ||
      static_cast<std::size_t>(R.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "R has wrong dimension (it should be " +
                        std::to_string(nu_) + "," + std::to_string(nu_) + ")");
  }
  R_ = R;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_N(const MatrixXs& N) {
  if (static_cast<std::size_t>(N.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(N.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "N has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  N_ = N;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_q(const VectorXs& q) {
  if (static_cast<std::size_t>(q.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "q has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  q_ = q;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_r(const VectorXs& r) {
  if (static_cast<std::size_t>(r.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "r has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  r_ = r;
}

}  // namespace crocoddyl
