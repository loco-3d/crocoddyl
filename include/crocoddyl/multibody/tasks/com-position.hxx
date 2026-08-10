///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
TaskModelCoMPositionTpl<Scalar>::TaskModelCoMPositionTpl(
    std::shared_ptr<StateMultibody> state, const Vector3s& cref,
    const std::size_t nu)
    : Base(state, 3, nu, true, true, true),
      cref_(cref),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
TaskModelCoMPositionTpl<Scalar>::TaskModelCoMPositionTpl(
    std::shared_ptr<StateMultibody> state, const Vector3s& cref)
    : Base(state, 3, state->get_nv(), true, true, true),
      cref_(cref),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
void TaskModelCoMPositionTpl<Scalar>::calc(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  d->com = d->pinocchio->com[0];
  data->y = d->com - cref_;

  d->vcom = d->pinocchio->vcom[0];
  data->v = d->vcom;
  if (data->compute_acceleration) {
    d->acom = d->pinocchio->acom[0];
    data->a = d->acom;
  } else {
    d->acom.setZero();
    data->a.setZero();
  }
}

template <typename Scalar>
void TaskModelCoMPositionTpl<Scalar>::calcDiff(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();

  data->Yx.leftCols(nv).noalias() = d->pinocchio->Jcom;
  pinocchio::getCenterOfMassVelocityDerivatives(*pin_model_.get(),
                                                *d->pinocchio, d->dvc_dq);
  data->Vx.leftCols(nv).noalias() = d->dvc_dq;
  data->Vx.rightCols(nv).noalias() = d->pinocchio->Jcom;
  if (!data->compute_acceleration) {
    d->dacom_dq.setZero();
    d->dacom_dv.setZero();
    return;
  }
  const Scalar inv_mass = Scalar(1) / d->pinocchio->mass[0];
  d->dacom_dq.noalias() = d->pinocchio->dFdq.template topRows<3>() * inv_mass;
  d->dacom_dv.noalias() = d->pinocchio->dFdv.template topRows<3>() * inv_mass;
  data->Ax.leftCols(nv).noalias() = d->dacom_dq;
  data->Ax.rightCols(nv).noalias() = d->dacom_dv;
  if (d->joint != nullptr) {
    data->Ax.noalias() += d->pinocchio->Jcom * d->joint->da_dx;
    data->Au.noalias() = d->pinocchio->Jcom * d->joint->da_du;
  }
}

template <typename Scalar>
std::shared_ptr<TaskDataAbstractTpl<Scalar>>
TaskModelCoMPositionTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
TaskModelCoMPositionTpl<NewScalar> TaskModelCoMPositionTpl<Scalar>::cast()
    const {
  typedef TaskModelCoMPositionTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      cref_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
const typename TaskModelCoMPositionTpl<Scalar>::Vector3s&
TaskModelCoMPositionTpl<Scalar>::get_reference() const {
  return cref_;
}

template <typename Scalar>
void TaskModelCoMPositionTpl<Scalar>::set_reference(const Vector3s& cref) {
  cref_ = cref;
}

template <typename Scalar>
void TaskModelCoMPositionTpl<Scalar>::print(std::ostream& os) const {
  typedef typename ScalarSelector<Scalar>::type PrintableScalar;
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "TaskModelCoMPosition {cref="
     << cref_.transpose().template cast<PrintableScalar>().format(fmt) << "}";
}

}  // namespace crocoddyl
