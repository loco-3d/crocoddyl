///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ResidualModelCoMPositionTpl<Scalar>::ResidualModelCoMPositionTpl(
    std::shared_ptr<StateMultibody> state, const Vector3s& cref,
    const std::size_t nu)
    : Base(state, 3, nu, true, false, false), cref_(cref) {}

template <typename Scalar>
ResidualModelCoMPositionTpl<Scalar>::ResidualModelCoMPositionTpl(
    std::shared_ptr<StateMultibody> state, const Vector3s& cref)
    : Base(state, 3, true, false, false), cref_(cref) {}

template <typename Scalar>
void ResidualModelCoMPositionTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  // Compute the residual residual give the reference CoMPosition position
  Data* d = static_cast<Data*>(data.get());
  data->r = d->pinocchio->com[0] - cref_;
}

template <typename Scalar>
void ResidualModelCoMPositionTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t nv = state_->get_nv();
  data->Rx.leftCols(nv) = d->pinocchio->Jcom;
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelCoMPositionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelCoMPositionTpl<NewScalar>
ResidualModelCoMPositionTpl<Scalar>::cast() const {
  typedef ResidualModelCoMPositionTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      cref_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelCoMPositionTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "ResidualModelCoMPosition {cref=" << cref_.transpose().format(fmt)
     << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s&
ResidualModelCoMPositionTpl<Scalar>::get_reference() const {
  return cref_;
}

template <typename Scalar>
void ResidualModelCoMPositionTpl<Scalar>::set_reference(const Vector3s& cref) {
  cref_ = cref;
}

}  // namespace crocoddyl
