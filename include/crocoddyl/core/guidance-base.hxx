///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
GuidanceModelAbstractTpl<Scalar>::GuidanceModelAbstractTpl(const std::size_t nr)
    : nr_(nr) {
  if (nr_ == 0) {
    throw_pretty("Invalid argument: guidance-model dimension must be positive");
  }
}

template <typename Scalar>
std::shared_ptr<typename GuidanceModelAbstractTpl<Scalar>::GuidanceDataAbstract>
GuidanceModelAbstractTpl<Scalar>::createData() const {
  return std::allocate_shared<GuidanceDataAbstract>(
      Eigen::aligned_allocator<GuidanceDataAbstract>(), this);
}

template <typename Scalar>
std::size_t GuidanceModelAbstractTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
void GuidanceModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << "GuidanceModelAbstract {nr=" << nr_ << "}";
}

template <typename Scalar>
void GuidanceModelAbstractTpl<Scalar>::checkErrorDimension(
    const Eigen::Ref<const VectorXs>& error) const {
  if (static_cast<std::size_t>(error.size()) != nr_) {
    throw_pretty("Invalid argument: error has wrong dimension ("
                 << error.size() << " provided, expected " << nr_ << ")");
  }
}

template <typename Scalar>
void GuidanceModelAbstractTpl<Scalar>::checkRateDimension(
    const Eigen::Ref<const VectorXs>& g) const {
  if (static_cast<std::size_t>(g.size()) != nr_) {
    throw_pretty("Invalid argument: g has wrong dimension ("
                 << g.size() << " provided, expected " << nr_ << ")");
  }
}

template <typename Scalar>
void GuidanceModelAbstractTpl<Scalar>::checkJacobianDimension(
    const Eigen::Ref<const MatrixXs>& Ge) const {
  if (static_cast<std::size_t>(Ge.rows()) != nr_ ||
      static_cast<std::size_t>(Ge.cols()) != nr_) {
    throw_pretty("Invalid argument: Ge has wrong dimension ("
                 << Ge.rows() << "x" << Ge.cols() << " provided, expected "
                 << nr_ << "x" << nr_ << ")");
  }
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const GuidanceModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
