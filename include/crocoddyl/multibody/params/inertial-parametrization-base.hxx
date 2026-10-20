///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
std::shared_ptr<typename InertialParametrizationAbstractTpl<
    Scalar>::InertialParametrizationDataAbstract>
InertialParametrizationAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<InertialParametrizationDataAbstract>(
      Eigen::aligned_allocator<InertialParametrizationDataAbstract>());
}

template <typename Scalar>
bool InertialParametrizationAbstractTpl<Scalar>::checkData(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data) const {
  return data != nullptr;
}

template <typename Scalar>
std::size_t InertialParametrizationAbstractTpl<Scalar>::get_np() const {
  return kDimension;
}

template <typename Scalar>
void InertialParametrizationAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
std::shared_ptr<InertialParametrizationBase>
InertialParametrizationAbstractTpl<Scalar>::cloneAsDouble() const {
  throw_pretty(
      "Invalid call: scalar casting is not implemented for this inertial "
      "parametrization");
}

template <typename Scalar>
std::shared_ptr<InertialParametrizationBase>
InertialParametrizationAbstractTpl<Scalar>::cloneAsFloat() const {
  throw_pretty(
      "Invalid call: scalar casting is not implemented for this inertial "
      "parametrization");
}

#ifdef CROCODDYL_WITH_CODEGEN
template <typename Scalar>
std::shared_ptr<InertialParametrizationBase>
InertialParametrizationAbstractTpl<Scalar>::cloneAsADDouble() const {
  throw_pretty(
      "Invalid call: scalar casting is not implemented for this inertial "
      "parametrization");
}

template <typename Scalar>
std::shared_ptr<InertialParametrizationBase>
InertialParametrizationAbstractTpl<Scalar>::cloneAsADFloat() const {
  throw_pretty(
      "Invalid call: scalar casting is not implemented for this inertial "
      "parametrization");
}
#endif

template <typename Scalar>
std::ostream& operator<<(
    std::ostream& os, const InertialParametrizationAbstractTpl<Scalar>& obj) {
  obj.print(os);
  return os;
}

}  // namespace crocoddyl
