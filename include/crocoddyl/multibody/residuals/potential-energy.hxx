///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/potential-energy.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelPotentialEnergyTpl<Scalar>::ResidualModelPotentialEnergyTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu,
    const std::size_t np, const Scalar U_ref, const std::string& param_name)
    : Base(state, 1, nu,
           /*q_dependent=*/true, /*v_dependent=*/false,
           /*u_dependent=*/false, np),
      U_ref_(U_ref),
      param_name_(param_name),
      pin_model_(internal::getMultibodyState<Scalar>(state)->get_pinocchio()) {}

template <typename Scalar>
void ResidualModelPotentialEnergyTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  if (d->inertial_data != nullptr) {
    internal::getDynamicsParamOffset<Scalar>(d->parameter_data, param_name_);
  }
  const std::size_t nq = state_->get_nq();
  data->r[0] = pinocchio::computePotentialEnergy(*pin_model_, *d->pinocchio,
                                                 x.head(nq)) -
               U_ref_;
}

template <typename Scalar>
void ResidualModelPotentialEnergyTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelPotentialEnergyTpl<Scalar>::calcDiff(
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
    const typename pinocchio::DataTpl<Scalar>::RowVectorXs& dU_dp =
        pinocchio::computePotentialEnergyRegressor(*pin_model_, *d->pinocchio,
                                                   x.head(nq));
    internal::addInertialEnergyRegressor<Scalar>(
        data.get(), d->inertial_data, dU_dp, d->np_offset, Scalar(1));
  }

  const VectorXs& g = pinocchio::computeGeneralizedGravity(
      *pin_model_, *d->pinocchio, x.head(nq));
  data->Rx.leftCols(nv).row(0) = g.transpose();
}

template <typename Scalar>
void ResidualModelPotentialEnergyTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelPotentialEnergyTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelPotentialEnergyTpl<NewScalar>
ResidualModelPotentialEnergyTpl<Scalar>::cast() const {
  typedef ResidualModelPotentialEnergyTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(), nu_, np_,
                 scalar_cast<NewScalar>(U_ref_), param_name_);
  return ret;
}

template <typename Scalar>
Scalar ResidualModelPotentialEnergyTpl<Scalar>::get_reference() const {
  return U_ref_;
}

template <typename Scalar>
void ResidualModelPotentialEnergyTpl<Scalar>::set_reference(
    const Scalar reference) {
  U_ref_ = reference;
}

template <typename Scalar>
const std::string& ResidualModelPotentialEnergyTpl<Scalar>::get_param_name()
    const {
  return param_name_;
}

template <typename Scalar>
void ResidualModelPotentialEnergyTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelPotentialEnergy {U_ref=" << U_ref_;
  if (!param_name_.empty()) {
    os << ", param=" << param_name_;
  }
  os << ", np=" << np_ << "}";
}

}  // namespace crocoddyl
