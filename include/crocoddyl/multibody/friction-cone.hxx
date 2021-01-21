///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <iostream>

namespace crocoddyl {

template <typename Scalar>
FrictionConeTpl<Scalar>::FrictionConeTpl()
    : nf_(4),
      A_(nf_ + 1, 3),
      ub_(nf_ + 1),
      lb_(nf_ + 1),
      R_(Matrix3s::Identity()),
      nsurf_(Vector3s::UnitZ()),
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
FrictionConeTpl<Scalar>::FrictionConeTpl(const Matrix3s& R, const Scalar mu, std::size_t nf, const bool inner_appr,
                                         const Scalar min_nforce, const Scalar max_nforce)
    : nf_(nf),
      R_(R),
      nsurf_(R_.transpose() * Vector3s::UnitZ()),
      mu_(mu),
      inner_appr_(inner_appr),
      min_nforce_(min_nforce),
      max_nforce_(max_nforce) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  A_ = MatrixX3s::Zero(nf_ + 1, 3);
  ub_ = VectorXs::Zero(nf_ + 1);
  lb_ = VectorXs::Zero(nf_ + 1);

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
FrictionConeTpl<Scalar>::FrictionConeTpl(const Vector3s& nsurf, const Scalar mu, std::size_t nf, const bool inner_appr,
                                         const Scalar min_nforce, const Scalar max_nforce)
    : nf_(nf),
      R_(Quaternions::FromTwoVectors(nsurf, Vector3s::UnitZ()).toRotationMatrix()),
      nsurf_(nsurf),
      mu_(mu),
      inner_appr_(inner_appr),
      min_nforce_(min_nforce),
      max_nforce_(max_nforce) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  if (!nsurf.isUnitary()) {
    nsurf_ /= nsurf.norm();
    std::cerr << "Warning: normal is not an unitary vector, then we normalized it" << std::endl;
  }
  A_ = MatrixX3s::Zero(nf_ + 1, 3);
  ub_ = VectorXs::Zero(nf_ + 1);
  lb_ = VectorXs::Zero(nf_ + 1);
  R_ = Quaternions::FromTwoVectors(nsurf_, Vector3s::UnitZ()).toRotationMatrix();

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
FrictionConeTpl<Scalar>::FrictionConeTpl(const FrictionConeTpl<Scalar>& cone)
    : nf_(cone.get_nf()),
      A_(cone.get_A()),
      ub_(cone.get_ub()),
      lb_(cone.get_lb()),
      R_(cone.get_R()),
      nsurf_(cone.get_nsurf()),
      mu_(cone.get_mu()),
      inner_appr_(cone.get_inner_appr()),
      min_nforce_(cone.get_min_nforce()),
      max_nforce_(cone.get_max_nforce()) {}

template <typename Scalar>
FrictionConeTpl<Scalar>::~FrictionConeTpl() {}

template <typename Scalar>
void FrictionConeTpl<Scalar>::update() {
  // Initialize the matrix and bounds
  A_.setZero();
  ub_.setZero();
  lb_.setOnes();
  lb_ *= -std::numeric_limits<Scalar>::max();

  // Compute the mu given the type of friction cone approximation
  Scalar mu = mu_;
  const Scalar theta = Scalar(2) * M_PI / static_cast<Scalar>(nf_);
  if (inner_appr_) {
    mu *= cos(theta / Scalar(2.));
  }

  // Update the inequality matrix and the bounds
  Scalar theta_i;
  Vector3s tsurf_i;
  for (std::size_t i = 0; i < nf_ / 2; ++i) {
    theta_i = theta * static_cast<Scalar>(i);
    tsurf_i << cos(theta_i), sin(theta_i), Scalar(0.);
    A_.row(2 * i) = (-mu * Vector3s::UnitZ() + tsurf_i).transpose() * R_;
    A_.row(2 * i + 1) = (-mu * Vector3s::UnitZ() - tsurf_i).transpose() * R_;
  }
  A_.row(nf_) = nsurf_.transpose();
  lb_(nf_) = min_nforce_;
  ub_(nf_) = max_nforce_;
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::update(const Vector3s& normal, const Scalar mu, const bool inner_appr,
                                     const Scalar min_nforce, const Scalar max_nforce) {
  set_nsurf(normal);
  set_mu(mu_);
  set_inner_appr(inner_appr);
  set_min_nforce(min_nforce);
  set_max_nforce(max_nforce);

  // Sanity checks
  if (!normal.isUnitary()) {
    nsurf_ /= normal.norm();
    std::cerr << "Warning: normal is not an unitary vector, then we normalized it" << std::endl;
  }
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0" << std::endl;
  }
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::max();
    std::cerr << "Warning: max_nforce has to be a positive value, set to maximum value" << std::endl;
  }

  update();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixX3s& FrictionConeTpl<Scalar>::get_A() const {
  return A_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& FrictionConeTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& FrictionConeTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s& FrictionConeTpl<Scalar>::get_R() const {
  return R_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s& FrictionConeTpl<Scalar>::get_nsurf() const {
  return nsurf_;
}

template <typename Scalar>
const Scalar FrictionConeTpl<Scalar>::get_mu() const {
  return mu_;
}

template <typename Scalar>
std::size_t FrictionConeTpl<Scalar>::get_nf() const {
  return nf_;
}

template <typename Scalar>
bool FrictionConeTpl<Scalar>::get_inner_appr() const {
  return inner_appr_;
}

template <typename Scalar>
const Scalar FrictionConeTpl<Scalar>::get_min_nforce() const {
  return min_nforce_;
}

template <typename Scalar>
const Scalar FrictionConeTpl<Scalar>::get_max_nforce() const {
  return max_nforce_;
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_R(const Matrix3s& R) {
  R_ = R;
  nsurf_ = R_.transpose() * Vector3s::UnitZ();
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_nsurf(const Vector3s& nsurf) {
  nsurf_ = nsurf;
  // Sanity checks
  if (!nsurf.isUnitary()) {
    nsurf_ /= nsurf.norm();
    std::cerr << "Warning: normal is not an unitary vector, then we normalized it" << std::endl;
  }
  R_ = Quaternions::FromTwoVectors(nsurf_, Vector3s::UnitZ()).toRotationMatrix();
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_mu(const Scalar mu) {
  mu_ = mu;
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_inner_appr(const bool inner_appr) {
  inner_appr_ = inner_appr;
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_min_nforce(const Scalar min_nforce) {
  min_nforce_ = min_nforce;
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_max_nforce(const Scalar max_nforce) {
  max_nforce_ = max_nforce;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const FrictionConeTpl<Scalar>& X) {
  os << "        R: " << X.get_R() << std::endl;
  os << "  (nsurf): " << X.get_nsurf().transpose() << std::endl;
  os << "        mu: " << X.get_mu() << std::endl;
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
