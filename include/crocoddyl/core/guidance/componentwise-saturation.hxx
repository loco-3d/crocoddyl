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
GuidanceModelComponentwiseSaturationTpl<
    Scalar>::GuidanceModelComponentwiseSaturationTpl(const VectorXs& gain,
                                                     const VectorXs& max_rate)
    : Base(static_cast<std::size_t>(gain.size())),
      gain_(gain),
      max_rate_(max_rate) {
  checkParameters(gain_, max_rate_);
}

template <typename Scalar>
void GuidanceModelComponentwiseSaturationTpl<Scalar>::calc(
    const std::shared_ptr<GuidanceDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& error) const {
  using std::tanh;

  checkErrorDimension(error);
  checkRateDimension(data->g);
  const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  for (std::size_t i = 0; i < nr_; ++i) {
    const Eigen::Index index = static_cast<Eigen::Index>(i);
    d->z[index] = gain_[index] * error[index] / max_rate_[index];
    d->tanh_z[index] = tanh(d->z[index]);
    d->g[index] = -max_rate_[index] * d->tanh_z[index];
  }
}

template <typename Scalar>
void GuidanceModelComponentwiseSaturationTpl<Scalar>::calcDiff(
    const std::shared_ptr<GuidanceDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& error) const {
  checkErrorDimension(error);
  checkJacobianDimension(data->Ge);
  const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  data->Ge.setZero();
  for (std::size_t i = 0; i < nr_; ++i) {
    const Eigen::Index index = static_cast<Eigen::Index>(i);
    data->Ge(index, index) =
        -gain_[index] * (Scalar(1) - d->tanh_z[index] * d->tanh_z[index]);
  }
}

template <typename Scalar>
template <typename NewScalar>
GuidanceModelComponentwiseSaturationTpl<NewScalar>
GuidanceModelComponentwiseSaturationTpl<Scalar>::cast() const {
  typedef GuidanceModelComponentwiseSaturationTpl<NewScalar> ReturnType;
  ReturnType ret(gain_.template cast<NewScalar>(),
                 max_rate_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
std::shared_ptr<typename GuidanceModelComponentwiseSaturationTpl<
    Scalar>::GuidanceDataAbstract>
GuidanceModelComponentwiseSaturationTpl<Scalar>::createData() const {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
const typename GuidanceModelComponentwiseSaturationTpl<Scalar>::VectorXs&
GuidanceModelComponentwiseSaturationTpl<Scalar>::get_gain() const {
  return gain_;
}

template <typename Scalar>
const typename GuidanceModelComponentwiseSaturationTpl<Scalar>::VectorXs&
GuidanceModelComponentwiseSaturationTpl<Scalar>::get_max_rate() const {
  return max_rate_;
}

template <typename Scalar>
void GuidanceModelComponentwiseSaturationTpl<Scalar>::set_gain(
    const VectorXs& gain) {
  checkParameters(gain, max_rate_);
  gain_ = gain;
}

template <typename Scalar>
void GuidanceModelComponentwiseSaturationTpl<Scalar>::set_max_rate(
    const VectorXs& max_rate) {
  checkParameters(gain_, max_rate);
  max_rate_ = max_rate;
}

template <typename Scalar>
void GuidanceModelComponentwiseSaturationTpl<Scalar>::print(
    std::ostream& os) const {
  os << "GuidanceModelComponentwiseSaturation {nr=" << nr_ << "}";
}

template <typename Scalar>
void GuidanceModelComponentwiseSaturationTpl<Scalar>::checkParameters(
    const VectorXs& gain, const VectorXs& max_rate) const {
  if (static_cast<std::size_t>(gain.size()) != nr_ ||
      static_cast<std::size_t>(max_rate.size()) != nr_) {
    throw_pretty(
        "Invalid argument: componentwise gain and max_rate must both have "
        "dimension "
        << nr_);
  }
  for (std::size_t i = 0; i < nr_; ++i) {
    const Eigen::Index index = static_cast<Eigen::Index>(i);
    if (gain[index] < Scalar(0)) {
      throw_pretty(
          "Invalid argument: componentwise guidance gains must be nonnegative");
    }
    if (max_rate[index] <= Scalar(0)) {
      throw_pretty(
          "Invalid argument: componentwise maximum rates must be positive");
    }
  }
}

}  // namespace crocoddyl
