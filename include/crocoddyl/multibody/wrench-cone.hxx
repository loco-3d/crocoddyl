///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <iostream>

namespace crocoddyl {

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl() : nf_(16) {
  A_.resize(nf_ + 1, 6);
  ub_.resize(nf_ + 1);
  lb_.resize(nf_ + 1);
  // compute the matrix
  update(Matrix3s::Identity(), Scalar(0.7), Vector2s(0.1, 0.05), Scalar(0.), std::numeric_limits<Scalar>::max());
}

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl(const Matrix3s& R, Scalar mu, const Vector2s& box_size, std::size_t nf,
                                     Scalar min_nforce, Scalar max_nforce)
    : nf_(nf) {
  if (nf_ % 2 != 0) {
    nf_ = 16;
    std::cerr << "Warning: nf has to be an even number, set to 16" << std::endl;
  }
  A_.resize(nf_ + 1, 6);
  ub_.resize(nf_ + 1);
  lb_.resize(nf_ + 1);

  // compute the matrix
  update(R, mu, box_size, min_nforce, max_nforce);
}

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl(const WrenchConeTpl<Scalar>& cone)
    : A_(cone.get_A()),
      ub_(cone.get_ub()),
      lb_(cone.get_lb()),
      R_(cone.get_R()),
      box_(cone.get_box()),
      mu_(cone.get_mu()),
      nf_(cone.get_nf()),
      min_nforce_(cone.get_min_nforce()),
      max_nforce_(cone.get_max_nforce()) {}

template <typename Scalar>
WrenchConeTpl<Scalar>::~WrenchConeTpl() {}

template <typename Scalar>
void WrenchConeTpl<Scalar>::update(const Matrix3s& R, Scalar mu, const Vector2s& box_size, Scalar min_nforce,
                                   Scalar max_nforce) {
  R_ = R;
  mu_ = mu;
  box_ = box_size;
  min_nforce_ = min_nforce;
  max_nforce_ = max_nforce;

  mu_ = mu_ / sqrt(Scalar(2.));
  A_.setZero();

  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0" << std::endl;
  }
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::max();
    std::cerr << "Warning: max_nforce has to be a positive value, set to maximun value" << std::endl;
  }

  A_.row(0).head(3) << Scalar(1.), Scalar(0.), -mu_;
  A_.row(1).head(3) << Scalar(-1.), Scalar(0.), -mu_;
  A_.row(2).head(3) << Scalar(0.), Scalar(1.), -mu_;
  A_.row(3).head(3) << Scalar(0.), Scalar(-1.), -mu_;
  A_.row(4).head(3) << Scalar(0.), Scalar(0.), Scalar(-1.);
  A_.row(5).segment(2, 3) << -box_(1), Scalar(1.), Scalar(0.);
  A_.row(6).segment(2, 3) << -box_(1), Scalar(-1.), Scalar(0.);
  A_.row(7).segment(2, 3) << -box_(0), Scalar(0.), Scalar(1.);
  A_.row(8).segment(2, 3) << -box_(0), Scalar(0.), Scalar(-1.);
  A_.row(9) << box_(1), box_(0), -mu_ * (box_(0) + box_(1)), -mu_, -mu_, Scalar(-1.);
  A_.row(10) << box_(1), -box_(0), -mu_ * (box_(0) + box_(1)), -mu_, mu_, Scalar(-1.);
  A_.row(11) << -box_(1), box_(0), -mu_ * (box_(0) + box_(1)), mu_, -mu_, Scalar(-1.);
  A_.row(12) << -box_(1), -box_(0), -mu_ * (box_(0) + box_(1)), mu_, mu_, Scalar(-1.);
  A_.row(13) << box_(1), box_(0), -mu_ * (box_(0) + box_(1)), mu_, mu_, Scalar(1.);
  A_.row(14) << box_(1), -box_(0), -mu_ * (box_(0) + box_(1)), mu_, -mu_, Scalar(1.);
  A_.row(15) << -box_(1), box_(0), -mu_ * (box_(0) + box_(1)), -mu_, mu_, Scalar(1.);
  A_.row(16) << -box_(1), -box_(0), -mu_ * (box_(0) + box_(1)), -mu_, -mu_, Scalar(1.);

  Matrix6s c_R_o = Matrix6s::Zero();
  c_R_o.topLeftCorner(3, 3) = R_.transpose();
  c_R_o.bottomRightCorner(3, 3) = R_.transpose();

  A_ = (A_ * c_R_o).eval();

  ub_.setZero();
  lb_.setOnes();
  lb_ *= -std::numeric_limits<Scalar>::max();
  ub_(4) = -min_nforce_;
  lb_(4) = -max_nforce_;
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
const typename MathBaseTpl<Scalar>::Matrix3s& WrenchConeTpl<Scalar>::get_R() const {
  return R_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& WrenchConeTpl<Scalar>::get_box() const {
  return box_;
}

template <typename Scalar>
Scalar WrenchConeTpl<Scalar>::get_mu() const {
  return mu_;
}

template <typename Scalar>
std::size_t WrenchConeTpl<Scalar>::get_nf() const {
  return nf_;
}

template <typename Scalar>
Scalar WrenchConeTpl<Scalar>::get_min_nforce() const {
  return min_nforce_;
}

template <typename Scalar>
Scalar WrenchConeTpl<Scalar>::get_max_nforce() const {
  return max_nforce_;
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_R(Matrix3s R) {
  update(R, mu_, box_, min_nforce_, max_nforce_);
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_box(Vector2s box) {
  update(R_, mu_, box, min_nforce_, max_nforce_);
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_mu(Scalar mu) {
  update(R_, mu, box_, min_nforce_, max_nforce_);
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_min_nforce(Scalar min_nforce) {
  update(R_, mu_, box_, min_nforce, max_nforce_);
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_max_nforce(Scalar max_nforce) {
  update(R_, mu_, box_, min_nforce_, max_nforce);
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const WrenchConeTpl<Scalar>& X) {
  os << "         R: " << X.get_R() << std::endl;
  os << "        mu: " << X.get_mu() << std::endl;
  os << "       box: " << X.get_box().transpose() << std::endl;
  os << "        nf: " << X.get_nf() << std::endl;
  os << " min_force: " << X.get_min_nforce() << std::endl;
  os << " max_force: " << X.get_max_nforce() << std::endl;
  return os;
}

}  // namespace crocoddyl
