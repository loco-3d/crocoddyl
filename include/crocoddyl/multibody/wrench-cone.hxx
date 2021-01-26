///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <iostream>

namespace crocoddyl {

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl()
    : nf_(4),
      A_(nf_ + 13, 6),
      ub_(nf_ + 13),
      lb_(nf_ + 13),
      R_(Matrix3s::Identity()),
      box_(std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max()),
      mu_(Scalar(0.7)),
      inner_appr_(true),
      min_nforce_(Scalar(0.)),
      max_nforce_(std::numeric_limits<Scalar>::max()) {
  A_.setZero();
  ub_.setZero();
  lb_.setZero();

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl(const Matrix3s& R, const Scalar mu, const Vector2s& box, const std::size_t nf,
                                     const bool inner_appr, const Scalar min_nforce, const Scalar max_nforce)
    : nf_(nf), R_(R), box_(box), mu_(mu), inner_appr_(inner_appr), min_nforce_(min_nforce), max_nforce_(max_nforce) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1." << std::endl;
  }
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0" << std::endl;
  }
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::max();
    std::cerr << "Warning: max_nforce has to be a positive value, set to maximum value" << std::endl;
  }
  A_ = MatrixX6s::Zero(nf_ + 13, 3);
  ub_ = VectorXs::Zero(nf_ + 13);
  lb_ = VectorXs::Zero(nf_ + 13);

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl(const Matrix3s& R, const Scalar mu, const Vector2s& box, std::size_t nf,
                                     const Scalar min_nforce, const Scalar max_nforce)
    : nf_(nf), R_(R), box_(box), mu_(mu), inner_appr_(true), min_nforce_(min_nforce), max_nforce_(max_nforce) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1." << std::endl;
  }
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0" << std::endl;
  }
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::max();
    std::cerr << "Warning: max_nforce has to be a positive value, set to maximum value" << std::endl;
  }
  A_ = MatrixX3s::Zero(nf_ + 13, 3);
  ub_ = VectorXs::Zero(nf_ + 13);
  lb_ = VectorXs::Zero(nf_ + 13);

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl(const WrenchConeTpl<Scalar>& cone)
    : nf_(cone.get_nf()),
      A_(cone.get_A()),
      ub_(cone.get_ub()),
      lb_(cone.get_lb()),
      R_(cone.get_R()),
      box_(cone.get_box()),
      mu_(cone.get_mu()),
      inner_appr_(cone.get_inner_appr()),
      min_nforce_(cone.get_min_nforce()),
      max_nforce_(cone.get_max_nforce()) {}

template <typename Scalar>
WrenchConeTpl<Scalar>::~WrenchConeTpl() {}

template <typename Scalar>
void WrenchConeTpl<Scalar>::update() {
  // Initialize the matrix and bounds
  A_.setZero();
  ub_.setZero();
  lb_.setOnes();
  lb_ *= -std::numeric_limits<Scalar>::max();

  // Compute the mu given the type of friction cone approximation
  Scalar mu = mu_;
  Scalar theta = Scalar(2) * M_PI / static_cast<Scalar>(nf_);
  if (inner_appr_) {
    mu *= cos(theta / Scalar(2.));
  }

  // Friction cone information
  // This segment of matrix is defined as
  // [ 1  0 -mu  0  0  0;
  //  -1  0 -mu  0  0  0;
  //   0  1 -mu  0  0  0;
  //   0 -1 -mu  0  0  0;
  //   0  0   1  0  0  0]
  for (std::size_t i = 0; i < nf_ / 2; ++i) {
    Scalar theta_i = theta * static_cast<Scalar>(i);
    Vector3s tsurf_i = Vector3s(cos(theta_i), sin(theta_i), Scalar(0.));
    Vector3s mu_nsurf = -mu * Vector3s::UnitZ();
    A_.row(2 * i).head(3) = (mu_nsurf + tsurf_i).transpose() * R_.transpose();
    A_.row(2 * i + 1).head(3) = (mu_nsurf - tsurf_i).transpose() * R_.transpose();
  }
  A_.row(nf_).head(3) = R_.col(2).transpose();
  lb_(nf_) = min_nforce_;
  ub_(nf_) = max_nforce_;

  // CoP information
  const Scalar L = box_(0) / Scalar(2.);
  const Scalar W = box_(1) / Scalar(2.);
  // This segment of matrix is defined as
  // [0 0 -W  1  0;
  //  0 0 -W -1  0;
  //  0 0 -L  0  1;
  //  0 0 -L  0 -1]
  A_.row(nf_ + 1) << -W * R_.col(2).transpose(), R_.col(0).transpose();
  A_.row(nf_ + 2) << -W * R_.col(2).transpose(), -R_.col(0).transpose();
  A_.row(nf_ + 3) << -L * R_.col(2).transpose(), R_.col(1).transpose();
  A_.row(nf_ + 4) << -L * R_.col(2).transpose(), -R_.col(1).transpose();

  // Yaw-tau information
  const Scalar mu_LW = -mu * (L + W);
  // The segment of the matrix that encodes the minimum torque is defined as
  // [ W  L -mu*(L+W) -mu -mu -1;
  //   W -L -mu*(L+W) -mu  mu -1;
  //  -W  L -mu*(L+W)  mu -mu -1;
  //  -W -L -mu*(L+W)  mu  mu -1]
  A_.row(nf_ + 5) << Vector3s(W, L, mu_LW).transpose() * R_.transpose(),
      Vector3s(-mu, -mu, Scalar(-1.)).transpose() * R_.transpose();
  A_.row(nf_ + 6) << Vector3s(W, -L, mu_LW).transpose() * R_.transpose(),
      Vector3s(-mu, mu, Scalar(-1.)).transpose() * R_.transpose();
  A_.row(nf_ + 7) << Vector3s(-W, L, mu_LW).transpose() * R_.transpose(),
      Vector3s(mu, -mu, Scalar(-1.)).transpose() * R_.transpose();
  A_.row(nf_ + 8) << Vector3s(-W, -L, mu_LW).transpose() * R_.transpose(),
      Vector3s(mu, mu, Scalar(-1.)).transpose() * R_.transpose();
  // The segment of the matrix that encodes the maximum torque is defined as
  // [ W  L -mu*(L+W)  mu  mu 1;
  //   W -L -mu*(L+W)  mu -mu 1;
  //  -W  L -mu*(L+W) -mu  mu 1;
  //  -W -L -mu*(L+W) -mu -mu 1]
  A_.row(nf_ + 9) << Vector3s(W, L, mu_LW).transpose() * R_.transpose(),
      Vector3s(mu, mu, Scalar(1.)).transpose() * R_.transpose();
  A_.row(nf_ + 10) << Vector3s(W, -L, mu_LW).transpose() * R_.transpose(),
      Vector3s(mu, -mu, Scalar(1.)).transpose() * R_.transpose();
  A_.row(nf_ + 11) << Vector3s(-W, L, mu_LW).transpose() * R_.transpose(),
      Vector3s(-mu, mu, Scalar(1.)).transpose() * R_.transpose();
  A_.row(nf_ + 12) << Vector3s(-W, -L, mu_LW).transpose() * R_.transpose(),
      Vector3s(-mu, -mu, Scalar(1.)).transpose() * R_.transpose();
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::update(const Matrix3s& R, const Scalar mu, const Vector2s& box, const Scalar min_nforce,
                                   const Scalar max_nforce) {
  set_R(R);
  set_mu(mu);
  set_inner_appr(inner_appr_);
  set_box(box);
  set_min_nforce(min_nforce);
  set_max_nforce(max_nforce);

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixX6s& WrenchConeTpl<Scalar>::get_A() const {
  return A_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& WrenchConeTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& WrenchConeTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
std::size_t WrenchConeTpl<Scalar>::get_nf() const {
  return nf_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s& WrenchConeTpl<Scalar>::get_R() const {
  return R_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& WrenchConeTpl<Scalar>::get_box() const {
  return box_;
}

template <typename Scalar>
const Scalar WrenchConeTpl<Scalar>::get_mu() const {
  return mu_;
}

template <typename Scalar>
bool WrenchConeTpl<Scalar>::get_inner_appr() const {
  return inner_appr_;
}

template <typename Scalar>
const Scalar WrenchConeTpl<Scalar>::get_min_nforce() const {
  return min_nforce_;
}

template <typename Scalar>
const Scalar WrenchConeTpl<Scalar>::get_max_nforce() const {
  return max_nforce_;
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_R(const Matrix3s& R) {
  R_ = R;
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_box(const Vector2s& box) {
  box_ = box;
  if (box_(0) < Scalar(0.)) {
    box_(0) = std::numeric_limits<Scalar>::max();
    std::cerr << "Warning: box(0) has to be a positive value, set to max. float" << std::endl;
  }
  if (box_(1) < Scalar(0.)) {
    box_(1) = std::numeric_limits<Scalar>::max();
    std::cerr << "Warning: box(0) has to be a positive value, set to max. float" << std::endl;
  }
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_mu(const Scalar mu) {
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1." << std::endl;
  }
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_inner_appr(const bool inner_appr) {
  inner_appr_ = inner_appr;
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_min_nforce(const Scalar min_nforce) {
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0" << std::endl;
  }
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_max_nforce(const Scalar max_nforce) {
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::max();
    std::cerr << "Warning: max_nforce has to be a positive value, set to maximum value" << std::endl;
  }
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const WrenchConeTpl<Scalar>& X) {
  os << "         R: " << X.get_R() << std::endl;
  os << "        mu: " << X.get_mu() << std::endl;
  os << "       box: " << X.get_box().transpose() << std::endl;
  os << "        nf: " << X.get_nf() << std::endl;
  os << "inner_appr: ";
  if (X.get_inner_appr()) {
    os << "true" << std::endl;
  } else {
    os << "false" << std::endl;
  }
  os << " min_force: " << X.get_min_nforce() << std::endl;
  os << " max_force: " << X.get_max_nforce() << std::endl;
  return os;
}

}  // namespace crocoddyl
