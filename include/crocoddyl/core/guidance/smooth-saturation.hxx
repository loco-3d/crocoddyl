///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <cmath>

namespace crocoddyl {

template <typename Scalar>
GuidanceModelSmoothSaturationTpl<Scalar>::GuidanceModelSmoothSaturationTpl(
    const std::size_t nr, const Scalar& gain, const Scalar& max_rate,
    const Scalar epsilon)
    : Base(nr), gain_(gain), max_rate_(max_rate), epsilon_(epsilon) {
  checkParameters(gain_, max_rate_, epsilon_);
}

template <typename Scalar>
void GuidanceModelSmoothSaturationTpl<Scalar>::calc(
    const std::shared_ptr<GuidanceDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& error) const {
  using std::sqrt;
  using std::tanh;

  checkErrorDimension(error);
  checkRateDimension(data->g);

  const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  const Scalar radius = sqrt(error.squaredNorm() + epsilon_ * epsilon_);
  const Scalar z = gain_ * radius / max_rate_;
  const Scalar tanh_z = tanh(z);
  const Scalar scale = max_rate_ * tanh_z / radius;
  d->radius = radius;
  d->z = z;
  d->tanh_z = tanh_z;
  d->scale = scale;
  d->g.noalias() = -scale * error;
}

template <typename Scalar>
void GuidanceModelSmoothSaturationTpl<Scalar>::calcDiff(
    const std::shared_ptr<GuidanceDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& error) const {
  checkErrorDimension(error);
  checkJacobianDimension(data->Ge);

  const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  const Scalar sech2_z = Scalar(1) - d->tanh_z * d->tanh_z;
  const Scalar radial_derivative =
      (gain_ * sech2_z - d->scale) / (d->radius * d->radius);

  d->Ge.setIdentity();
  d->Ge *= -d->scale;
  d->Ge.noalias() -= radial_derivative * error * error.transpose();
}

template <typename Scalar>
template <typename NewScalar>
GuidanceModelSmoothSaturationTpl<NewScalar>
GuidanceModelSmoothSaturationTpl<Scalar>::cast() const {
  typedef GuidanceModelSmoothSaturationTpl<NewScalar> ReturnType;
  ReturnType ret(nr_, scalar_cast<NewScalar>(gain_),
                 scalar_cast<NewScalar>(max_rate_),
                 scalar_cast<NewScalar>(epsilon_));
  return ret;
}

template <typename Scalar>
std::shared_ptr<
    typename GuidanceModelSmoothSaturationTpl<Scalar>::GuidanceDataAbstract>
GuidanceModelSmoothSaturationTpl<Scalar>::createData() const {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
const Scalar& GuidanceModelSmoothSaturationTpl<Scalar>::get_gain() const {
  return gain_;
}

template <typename Scalar>
const Scalar& GuidanceModelSmoothSaturationTpl<Scalar>::get_max_rate() const {
  return max_rate_;
}

template <typename Scalar>
const Scalar& GuidanceModelSmoothSaturationTpl<Scalar>::get_epsilon() const {
  return epsilon_;
}

template <typename Scalar>
void GuidanceModelSmoothSaturationTpl<Scalar>::set_gain(const Scalar& gain) {
  checkParameters(gain, max_rate_, epsilon_);
  gain_ = gain;
}

template <typename Scalar>
void GuidanceModelSmoothSaturationTpl<Scalar>::set_max_rate(
    const Scalar& max_rate) {
  checkParameters(gain_, max_rate, epsilon_);
  max_rate_ = max_rate;
}

template <typename Scalar>
void GuidanceModelSmoothSaturationTpl<Scalar>::set_epsilon(
    const Scalar& epsilon) {
  checkParameters(gain_, max_rate_, epsilon);
  epsilon_ = epsilon;
}

template <typename Scalar>
void GuidanceModelSmoothSaturationTpl<Scalar>::print(std::ostream& os) const {
  os << "GuidanceModelSmoothSaturation {nr=" << nr_ << "}";
}

template <typename Scalar>
void GuidanceModelSmoothSaturationTpl<Scalar>::checkParameters(
    const Scalar& gain, const Scalar& max_rate, const Scalar& epsilon) const {
  if (gain < Scalar(0)) {
    throw_pretty("Invalid argument: guidance gain must be nonnegative");
  }
  if (max_rate <= Scalar(0)) {
    throw_pretty("Invalid argument: maximum desired rate must be positive");
  }
  if (epsilon <= Scalar(0)) {
    throw_pretty("Invalid argument: smoothing epsilon must be positive");
  }
}

}  // namespace crocoddyl
