///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <iostream>

namespace crocoddyl {

template <typename Scalar>
FrictionConeTpl<Scalar>::FrictionConeTpl() : nf_(4) {
  A_.resize(nf_ + 1, 3);
  lb_.resize(nf_ + 1);
  ub_.resize(nf_ + 1);
  // compute the matrix
  update(Vector3s(0, 0, 1), Scalar(0.7), true, Scalar(0.), std::numeric_limits<Scalar>::max());
}

template <typename Scalar>
FrictionConeTpl<Scalar>::FrictionConeTpl(const Vector3s& normal, const Scalar& mu, std::size_t nf, bool inner_appr,
                                         const Scalar& min_nforce, const Scalar& max_nforce)
    : nf_(nf) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  A_.resize(nf_ + 1, 3);
  lb_.resize(nf_ + 1);
  ub_.resize(nf_ + 1);

  // compute the matrix
  update(normal, mu, inner_appr, min_nforce, max_nforce);
}

template <typename Scalar>
FrictionConeTpl<Scalar>::FrictionConeTpl(const FrictionConeTpl<Scalar>& cone)
    : A_(cone.get_A()),
      lb_(cone.get_lb()),
      ub_(cone.get_ub()),
      nsurf_(cone.get_nsurf()),
      mu_(cone.get_mu()),
      nf_(cone.get_nf()),
      inner_appr_(cone.get_inner_appr()),
      min_nforce_(cone.get_min_nforce()),
      max_nforce_(cone.get_max_nforce()) {}

template <typename Scalar>
FrictionConeTpl<Scalar>::~FrictionConeTpl() {}

template <typename Scalar>
void FrictionConeTpl<Scalar>::update(const Vector3s& normal, const Scalar& mu, bool inner_appr,
                                     const Scalar& min_nforce, const Scalar& max_nforce) {
  nsurf_ = normal;
  mu_ = mu;
  inner_appr_ = inner_appr;
  min_nforce_ = min_nforce;
  max_nforce_ = max_nforce;

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
    std::cerr << "Warning: max_nforce has to be a positive value, set to maximun value" << std::endl;
  }

  Scalar theta = Scalar(2) * M_PI / static_cast<Scalar>(nf_);
  if (inner_appr_) {
    mu_ *= cos(theta / Scalar(2.));
  }

  Matrix3s c_R_o = Quaternions::FromTwoVectors(nsurf_, Vector3s::UnitZ()).toRotationMatrix();
  for (std::size_t i = 0; i < nf_ / 2; ++i) {
    Scalar theta_i = theta * static_cast<Scalar>(i);
    Vector3s tsurf_i = Vector3s(cos(theta_i), sin(theta_i), 0.);
    A_.row(2 * i) = (-mu_ * Vector3s::UnitZ() + tsurf_i).transpose() * c_R_o;
    A_.row(2 * i + 1) = (-mu_ * Vector3s::UnitZ() - tsurf_i).transpose() * c_R_o;
    lb_(2 * i) = -std::numeric_limits<Scalar>::max();
    lb_(2 * i + 1) = -std::numeric_limits<Scalar>::max();
    ub_(2 * i) = 0.;
    ub_(2 * i + 1) = Scalar(0.);
  }
  A_.row(nf_) = nsurf_.transpose();
  lb_(nf_) = min_nforce_;
  ub_(nf_) = max_nforce_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixX3s& FrictionConeTpl<Scalar>::get_A() const {
  return A_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& FrictionConeTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& FrictionConeTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s& FrictionConeTpl<Scalar>::get_nsurf() const {
  return nsurf_;
}

template <typename Scalar>
const Scalar& FrictionConeTpl<Scalar>::get_mu() const {
  return mu_;
}

template <typename Scalar>
const std::size_t& FrictionConeTpl<Scalar>::get_nf() const {
  return nf_;
}

template <typename Scalar>
const bool& FrictionConeTpl<Scalar>::get_inner_appr() const {
  return inner_appr_;
}

template <typename Scalar>
const Scalar& FrictionConeTpl<Scalar>::get_min_nforce() const {
  return min_nforce_;
}

template <typename Scalar>
const Scalar& FrictionConeTpl<Scalar>::get_max_nforce() const {
  return max_nforce_;
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_nsurf(Vector3s nsurf) {
  update(nsurf, mu_, inner_appr_, min_nforce_, max_nforce_);
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_mu(Scalar mu) {
  update(nsurf_, mu, inner_appr_, min_nforce_, max_nforce_);
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_inner_appr(bool inner_appr) {
  update(nsurf_, mu_, inner_appr, min_nforce_, max_nforce_);
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_min_nforce(Scalar min_nforce) {
  update(nsurf_, mu_, inner_appr_, min_nforce, max_nforce_);
}

template <typename Scalar>
void FrictionConeTpl<Scalar>::set_max_nforce(Scalar max_nforce) {
  update(nsurf_, mu_, inner_appr_, min_nforce_, max_nforce);
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const FrictionConeTpl<Scalar>& X) {
  os << "    normal: " << X.get_nsurf().transpose() << std::endl;
  os << "        mu: " << X.get_mu() << std::endl;
  os << "        nf: " << X.get_nf() << std::endl;
  os << "inner_appr: " << X.get_inner_appr() << std::endl;
  os << " min_force: " << X.get_min_nforce() << std::endl;
  os << " max_force: " << X.get_max_nforce() << std::endl;
  return os;
}

}  // namespace crocoddyl
