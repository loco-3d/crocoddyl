///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
GuidanceModelLinearTpl<Scalar>::GuidanceModelLinearTpl(const MatrixXs& gain)
    : Base(static_cast<std::size_t>(gain.rows())), gain_(gain) {
  if (gain_.rows() != gain_.cols()) {
    throw_pretty("Invalid argument: linear guidance gain must be square");
  }
}

template <typename Scalar>
GuidanceModelLinearTpl<Scalar>::GuidanceModelLinearTpl(
    const VectorXs& diagonal_gain)
    : Base(static_cast<std::size_t>(diagonal_gain.size())),
      gain_(diagonal_gain.asDiagonal()) {}

template <typename Scalar>
GuidanceModelLinearTpl<Scalar>::GuidanceModelLinearTpl(const std::size_t nr,
                                                       const Scalar& gain)
    : Base(nr), gain_(MatrixXs::Identity(nr, nr) * gain) {}

template <typename Scalar>
void GuidanceModelLinearTpl<Scalar>::calc(
    const std::shared_ptr<GuidanceDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& error) const {
  checkErrorDimension(error);
  checkRateDimension(data->g);
  data->g.noalias() = -gain_ * error;
}

template <typename Scalar>
void GuidanceModelLinearTpl<Scalar>::calcDiff(
    const std::shared_ptr<GuidanceDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& error) const {
  checkErrorDimension(error);
  checkJacobianDimension(data->Ge);
  data->Ge = -gain_;
}

template <typename Scalar>
template <typename NewScalar>
GuidanceModelLinearTpl<NewScalar> GuidanceModelLinearTpl<Scalar>::cast() const {
  typedef GuidanceModelLinearTpl<NewScalar> ReturnType;
  typename ReturnType::MatrixXs gain = gain_.template cast<NewScalar>();
  ReturnType ret(gain);
  return ret;
}

template <typename Scalar>
const typename GuidanceModelLinearTpl<Scalar>::MatrixXs&
GuidanceModelLinearTpl<Scalar>::get_gain() const {
  return gain_;
}

template <typename Scalar>
void GuidanceModelLinearTpl<Scalar>::set_gain(const MatrixXs& gain) {
  if (static_cast<std::size_t>(gain.rows()) != nr_ ||
      static_cast<std::size_t>(gain.cols()) != nr_) {
    throw_pretty("Invalid argument: gain has wrong dimension ("
                 << gain.rows() << "x" << gain.cols() << " provided, expected "
                 << nr_ << "x" << nr_ << ")");
  }
  gain_ = gain;
}

template <typename Scalar>
void GuidanceModelLinearTpl<Scalar>::print(std::ostream& os) const {
  os << "GuidanceModelLinear {nr=" << nr_ << "}";
}

}  // namespace crocoddyl
