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
ResidualModelCentroidalMomentumTpl<Scalar>::ResidualModelCentroidalMomentumTpl(
    std::shared_ptr<StateMultibody> state, const Vector6s& href,
    const std::size_t nu)
    : Base(state, 6, nu, true, true, false),
      href_(href),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelCentroidalMomentumTpl<Scalar>::ResidualModelCentroidalMomentumTpl(
    std::shared_ptr<StateMultibody> state, const Vector6s& href)
    : Base(state, 6, true, true, false),
      href_(href),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
void ResidualModelCentroidalMomentumTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  // Compute the residual residual give the reference centroidal momentum
  Data* d = static_cast<Data*>(data.get());
  data->r = d->pinocchio->hg.toVector() - href_;
}

template <typename Scalar>
void ResidualModelCentroidalMomentumTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t& nv = state_->get_nv();
  Eigen::Ref<Matrix6xs> Rq(data->Rx.leftCols(nv));
  Eigen::Ref<Matrix6xs> Rv(data->Rx.rightCols(nv));
  pinocchio::getCentroidalDynamicsDerivatives(*pin_model_.get(), *d->pinocchio,
                                              Rq, d->dhd_dq, d->dhd_dv, Rv);
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelCentroidalMomentumTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelCentroidalMomentumTpl<NewScalar>
ResidualModelCentroidalMomentumTpl<Scalar>::cast() const {
  typedef ResidualModelCentroidalMomentumTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      href_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelCentroidalMomentumTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "ResidualModelCentroidalMomentum {href="
     << href_.transpose().format(fmt) << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector6s&
ResidualModelCentroidalMomentumTpl<Scalar>::get_reference() const {
  return href_;
}

template <typename Scalar>
void ResidualModelCentroidalMomentumTpl<Scalar>::set_reference(
    const Vector6s& href) {
  href_ = href;
}

}  // namespace crocoddyl
