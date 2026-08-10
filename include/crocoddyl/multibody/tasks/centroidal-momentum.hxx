///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
TaskModelCentroidalMomentumTpl<Scalar>::TaskModelCentroidalMomentumTpl(
    std::shared_ptr<StateMultibody> state, const Vector6s& href,
    const Vector6s& hdot_ref, const std::size_t nu)
    : Base(state, 6, nu, true, true, true, false),
      href_(href),
      hdot_ref_(hdot_ref),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
TaskModelCentroidalMomentumTpl<Scalar>::TaskModelCentroidalMomentumTpl(
    std::shared_ptr<StateMultibody> state, const Vector6s& href,
    const Vector6s& hdot_ref)
    : TaskModelCentroidalMomentumTpl(state, href, hdot_ref, state->get_nv()) {}

template <typename Scalar>
TaskModelCentroidalMomentumTpl<Scalar>::TaskModelCentroidalMomentumTpl(
    std::shared_ptr<StateMultibody> state, const Vector6s& href,
    const std::size_t nu)
    : TaskModelCentroidalMomentumTpl(state, href, Vector6s::Zero(), nu) {}

template <typename Scalar>
TaskModelCentroidalMomentumTpl<Scalar>::TaskModelCentroidalMomentumTpl(
    std::shared_ptr<StateMultibody> state, const Vector6s& href)
    : TaskModelCentroidalMomentumTpl(state, href, Vector6s::Zero(),
                                     state->get_nv()) {}

template <typename Scalar>
void TaskModelCentroidalMomentumTpl<Scalar>::calc(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  data->y = d->pinocchio->hg.toVector();
  data->y.noalias() -= href_;

  // The momentum rate is a Pinocchio kinematic quantity. Reading it from the
  // shared cache keeps this task usable when no JointData is supplied.
  d->hdot = d->pinocchio->dhg.toVector();
  data->v = d->hdot;
  data->v.noalias() -= hdot_ref_;

  data->a.setZero();
}

template <typename Scalar>
void TaskModelCentroidalMomentumTpl<Scalar>::calcDiff(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();

  pinocchio::getCentroidalDynamicsDerivatives(*pin_model_.get(), *d->pinocchio,
                                              d->dh_dq, d->dhdot_dq,
                                              d->dhdot_dv, d->dhdot_da);
  data->Yx.leftCols(nv) = d->dh_dq;
  data->Yx.rightCols(nv) = d->dhdot_da;
  data->Vx.leftCols(nv) = d->dhdot_dq;
  data->Vx.rightCols(nv) = d->dhdot_dv;
  data->Yu.setZero();
  data->Vu.setZero();
  if (d->joint != nullptr) {
    data->Vx.noalias() += d->dhdot_da * d->joint->da_dx;
    data->Vu.noalias() = d->dhdot_da * d->joint->da_du;
  }
}

template <typename Scalar>
std::shared_ptr<TaskDataAbstractTpl<Scalar>>
TaskModelCentroidalMomentumTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
TaskModelCentroidalMomentumTpl<NewScalar>
TaskModelCentroidalMomentumTpl<Scalar>::cast() const {
  typedef TaskModelCentroidalMomentumTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  return ReturnType(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      href_.template cast<NewScalar>(), hdot_ref_.template cast<NewScalar>(),
      nu_);
}

template <typename Scalar>
const typename TaskModelCentroidalMomentumTpl<Scalar>::Vector6s&
TaskModelCentroidalMomentumTpl<Scalar>::get_reference() const {
  return href_;
}

template <typename Scalar>
const typename TaskModelCentroidalMomentumTpl<Scalar>::Vector6s&
TaskModelCentroidalMomentumTpl<Scalar>::get_rate_reference() const {
  return hdot_ref_;
}

template <typename Scalar>
void TaskModelCentroidalMomentumTpl<Scalar>::set_reference(
    const Vector6s& href) {
  href_ = href;
}

template <typename Scalar>
void TaskModelCentroidalMomentumTpl<Scalar>::set_rate_reference(
    const Vector6s& hdot_ref) {
  hdot_ref_ = hdot_ref;
}

template <typename Scalar>
void TaskModelCentroidalMomentumTpl<Scalar>::print(std::ostream& os) const {
  typedef typename ScalarSelector<Scalar>::type PrintableScalar;
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "TaskModelCentroidalMomentum {href="
     << href_.transpose().template cast<PrintableScalar>().format(fmt)
     << ", hdot_ref="
     << hdot_ref_.transpose().template cast<PrintableScalar>().format(fmt)
     << "}";
}

}  // namespace crocoddyl
