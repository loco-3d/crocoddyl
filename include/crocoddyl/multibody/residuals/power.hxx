///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/power.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelPowerTpl<Scalar>::ResidualModelPowerTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu,
    const std::size_t np, const Scalar P_ref,
    const std::string& inertial_param_name,
    const std::string& actuation_param_name)
    : Base(state, 1, nu,
           /*q_dependent=*/true, /*v_dependent=*/true,
           /*u_dependent=*/true, np),
      P_ref_(P_ref),
      inertial_param_name_(inertial_param_name),
      actuation_param_name_(actuation_param_name),
      pin_model_(internal::getMultibodyState<Scalar>(state)->get_pinocchio()) {}

template <typename Scalar>
void ResidualModelPowerTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  data->r.setZero();
  if (d->observer == nullptr || d->observer->xnext == nullptr) {
    return;
  }
  if (d->inertial_data != nullptr) {
    internal::getDynamicsParamOffset<Scalar>(d->parameter_data,
                                             inertial_param_name_);
  }
  if (d->actuation_data != nullptr) {
    internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
        *d->parameter_data, actuation_param_name_);
  }

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs> > qm = x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs> > vm =
      x.segment(nq, nv);
  const Eigen::VectorBlock<VectorXs, Eigen::Dynamic> qp =
      d->observer->xnext->head(nq);
  const Eigen::VectorBlock<VectorXs, Eigen::Dynamic> vp =
      d->observer->xnext->segment(nq, nv);

  const Scalar Em =
      pinocchio::computeMechanicalEnergy(*pin_model_, *d->pinocchio, qm, vm);
  const Scalar Ep =
      pinocchio::computeMechanicalEnergy(*pin_model_, *d->pinocchio, qp, vp);
  data->r[0] = Ep - Em - P_ref_;
  if (!actuation_param_name_.empty()) {
    data->r[0] += (*d->observer->dissipative_E)[0];
  }
}

template <typename Scalar>
void ResidualModelPowerTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->r.setZero();
}

template <typename Scalar>
void ResidualModelPowerTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  data->Rx.setZero();
  data->Ru.setZero();
  data->Rp.setZero();
  if (d->observer == nullptr || !d->observer->hasObserverData()) {
    return;
  }
  if (d->actuation_data != nullptr) {
    internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
        *d->parameter_data, actuation_param_name_);
  }

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs> > qm = x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs> > vm =
      x.segment(nq, nv);
  const Eigen::VectorBlock<VectorXs, Eigen::Dynamic> qp =
      d->observer->xnext->head(nq);
  const Eigen::VectorBlock<VectorXs, Eigen::Dynamic> vp =
      d->observer->xnext->segment(nq, nv);

  internal::computeKineticEnergyJacobian<Scalar>(*pin_model_, *d->pinocchio, qm,
                                                 vm, d->anone, d->J, d->Jout,
                                                 d->dV_dqv, d->dK_dqv, d->tmp6);
  d->dTm_dqv = d->dK_dqv;
  internal::computeKineticEnergyJacobian<Scalar>(*pin_model_, *d->pinocchio, qp,
                                                 vp, d->anone, d->J, d->Jout,
                                                 d->dV_dqv, d->dK_dqv, d->tmp6);
  d->dTp_dqv = d->dK_dqv;

  d->dUp_dq =
      pinocchio::computeGeneralizedGravity(*pin_model_, *d->pinocchio, qp);
  d->dUm_dq =
      pinocchio::computeGeneralizedGravity(*pin_model_, *d->pinocchio, qm);

  d->dEp_dx.setZero();
  d->dEp_dx.head(nv) = d->dUp_dq;
  d->dEp_dx += d->dTp_dqv;

  if (d->inertial_data != nullptr) {
    d->inertial_np_offset = internal::getDynamicsParamOffset<Scalar>(
        d->parameter_data, inertial_param_name_);
    const typename pinocchio::DataTpl<Scalar>::RowVectorXs& dT_dp_p =
        pinocchio::computeKineticEnergyRegressor(*pin_model_, *d->pinocchio, qp,
                                                 vp);
    internal::addInertialEnergyRegressor<Scalar>(data.get(), d->inertial_data,
                                                 dT_dp_p, d->inertial_np_offset,
                                                 Scalar(1));
    const typename pinocchio::DataTpl<Scalar>::RowVectorXs& dU_dp_p =
        pinocchio::computePotentialEnergyRegressor(*pin_model_, *d->pinocchio,
                                                   qp);
    internal::addInertialEnergyRegressor<Scalar>(data.get(), d->inertial_data,
                                                 dU_dp_p, d->inertial_np_offset,
                                                 Scalar(1));

    const typename pinocchio::DataTpl<Scalar>::RowVectorXs& dT_dp_m =
        pinocchio::computeKineticEnergyRegressor(*pin_model_, *d->pinocchio, qm,
                                                 vm);
    internal::addInertialEnergyRegressor<Scalar>(data.get(), d->inertial_data,
                                                 dT_dp_m, d->inertial_np_offset,
                                                 Scalar(-1));
    const typename pinocchio::DataTpl<Scalar>::RowVectorXs& dU_dp_m =
        pinocchio::computePotentialEnergyRegressor(*pin_model_, *d->pinocchio,
                                                   qm);
    internal::addInertialEnergyRegressor<Scalar>(data.get(), d->inertial_data,
                                                 dU_dp_m, d->inertial_np_offset,
                                                 Scalar(-1));
  }

  data->Rp.noalias() += d->dEp_dx.transpose() * (*d->observer->int_Fp);

  if (d->actuation_data != nullptr) {
    data->Rp += *d->observer->Ep;
  }

  data->Rx.noalias() = d->dUp_dq.transpose() * d->observer->int_Fx->topRows(nv);
  data->Rx.leftCols(nv).row(0) -= d->dUm_dq.transpose();
  data->Ru.noalias() = d->dUp_dq.transpose() * d->observer->int_Fu->topRows(nv);
  data->Rx.noalias() += d->dTp_dqv.transpose() * (*d->observer->int_Fx);
  data->Ru.noalias() += d->dTp_dqv.transpose() * (*d->observer->int_Fu);
  data->Rx.row(0) -= d->dTm_dqv.transpose();
  if (d->actuation_data != nullptr) {
    data->Rx += *d->observer->Ex;
    data->Ru += *d->observer->Eu;
  }
}

template <typename Scalar>
void ResidualModelPowerTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->Rx.setZero();
  data->Ru.setZero();
  data->Rp.setZero();
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelPowerTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelPowerTpl<NewScalar> ResidualModelPowerTpl<Scalar>::cast() const {
  typedef ResidualModelPowerTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(), nu_, np_,
                 scalar_cast<NewScalar>(P_ref_), inertial_param_name_,
                 actuation_param_name_);
  return ret;
}

template <typename Scalar>
Scalar ResidualModelPowerTpl<Scalar>::get_reference() const {
  return P_ref_;
}

template <typename Scalar>
void ResidualModelPowerTpl<Scalar>::set_reference(const Scalar reference) {
  P_ref_ = reference;
}

template <typename Scalar>
const std::string& ResidualModelPowerTpl<Scalar>::get_inertial_param_name()
    const {
  return inertial_param_name_;
}

template <typename Scalar>
const std::string& ResidualModelPowerTpl<Scalar>::get_actuation_param_name()
    const {
  return actuation_param_name_;
}

template <typename Scalar>
void ResidualModelPowerTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelPower {P_ref=" << P_ref_;
  if (!inertial_param_name_.empty()) {
    os << ", inertial_param=" << inertial_param_name_;
  }
  if (!actuation_param_name_.empty()) {
    os << ", actuation_param=" << actuation_param_name_;
  }
  os << ", np=" << np_ << "}";
}

}  // namespace crocoddyl
