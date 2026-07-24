///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/kinetic-energy.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelKineticEnergyTpl<Scalar>::ResidualModelKineticEnergyTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu,
    const std::size_t np, const Scalar T_ref, const std::string& param_name)
    : Base(state, 1, nu,
           /*q_dependent=*/true, /*v_dependent=*/true,
           /*u_dependent=*/false, np),
      T_ref_(T_ref),
      param_name_(param_name),
      pin_model_(internal::getMultibodyState<Scalar>(state)->get_pinocchio()) {}

template <typename Scalar>
void ResidualModelKineticEnergyTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  if (d->inertial_data != nullptr) {
    internal::getDynamicsParamOffset<Scalar>(d->parameter_data, param_name_);
  }
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  data->r[0] = pinocchio::computeKineticEnergy(*pin_model_, *d->pinocchio,
                                               x.head(nq), x.segment(nq, nv)) -
               T_ref_;
}

template <typename Scalar>
void ResidualModelKineticEnergyTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelKineticEnergyTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  data->Rx.setZero();
  data->Ru.setZero();
  data->Rp.setZero();

  if (d->inertial_data != nullptr) {
    d->np_offset = internal::getDynamicsParamOffset<Scalar>(d->parameter_data,
                                                            param_name_);
    const typename pinocchio::DataTpl<Scalar>::RowVectorXs& dT_dp =
        pinocchio::computeKineticEnergyRegressor(*pin_model_, *d->pinocchio,
                                                 x.head(nq), x.segment(nq, nv));
    internal::addInertialEnergyRegressor<Scalar>(
        data.get(), d->inertial_data, dT_dp, d->np_offset, Scalar(1));
  }

  internal::computeKineticEnergyJacobian<Scalar>(
      *pin_model_, *d->pinocchio, x.head(nq), x.segment(nq, nv), d->J, d->Jout,
      d->dV_dqv, d->dT_dqv, d->tmp6);
  data->Rx.row(0) = d->dT_dqv.transpose();
}

template <typename Scalar>
void ResidualModelKineticEnergyTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelKineticEnergyTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelKineticEnergyTpl<NewScalar>
ResidualModelKineticEnergyTpl<Scalar>::cast() const {
  typedef ResidualModelKineticEnergyTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(), nu_, np_,
                 scalar_cast<NewScalar>(T_ref_), param_name_);
  return ret;
}

template <typename Scalar>
Scalar ResidualModelKineticEnergyTpl<Scalar>::get_reference() const {
  return T_ref_;
}

template <typename Scalar>
void ResidualModelKineticEnergyTpl<Scalar>::set_reference(
    const Scalar reference) {
  T_ref_ = reference;
}

template <typename Scalar>
const std::string& ResidualModelKineticEnergyTpl<Scalar>::get_param_name()
    const {
  return param_name_;
}

template <typename Scalar>
void ResidualModelKineticEnergyTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelKineticEnergy {T_ref=" << T_ref_;
  if (!param_name_.empty()) {
    os << ", param=" << param_name_;
  }
  os << ", np=" << np_ << "}";
}

}  // namespace crocoddyl
