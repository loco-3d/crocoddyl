///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
TaskModelJointPositionTpl<Scalar>::TaskModelJointPositionTpl(
    std::shared_ptr<StateMultibody> state, const VectorXs& xref,
    const VectorXs& aref, const std::size_t nu)
    : Base(state, state->get_nv(), nu, true, true, true, true),
      xref_(xref),
      aref_(aref) {}

template <typename Scalar>
TaskModelJointPositionTpl<Scalar>::TaskModelJointPositionTpl(
    std::shared_ptr<StateMultibody> state, const VectorXs& xref,
    const VectorXs& aref)
    : TaskModelJointPositionTpl(state, xref, aref, state->get_nv()) {}

template <typename Scalar>
TaskModelJointPositionTpl<Scalar>::TaskModelJointPositionTpl(
    std::shared_ptr<StateMultibody> state, const VectorXs& xref,
    const std::size_t nu)
    : TaskModelJointPositionTpl(state, xref, VectorXs::Zero(state->get_nv()),
                                nu) {}

template <typename Scalar>
TaskModelJointPositionTpl<Scalar>::TaskModelJointPositionTpl(
    std::shared_ptr<StateMultibody> state, const VectorXs& xref)
    : TaskModelJointPositionTpl(state, xref, VectorXs::Zero(state->get_nv()),
                                state->get_nv()) {}

template <typename Scalar>
void TaskModelJointPositionTpl<Scalar>::calc(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();

  state_->diff(xref_, x, d->dx);
  data->y = d->dx.head(nv);
  data->v = d->dx.tail(nv);
  if (data->compute_acceleration &&
      (d->joint != nullptr or d->pinocchio != nullptr)) {
    data->a = d->joint != nullptr ? d->joint->a : d->pinocchio->ddq;
    data->a.noalias() -= aref_;
  } else {
    data->a.setZero();
  }
}

template <typename Scalar>
void TaskModelJointPositionTpl<Scalar>::calcDiff(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();

  state_->Jdiff(xref_, x, d->Jdiff, d->Jdiff, second);
  data->Yx = d->Jdiff.topRows(nv);
  data->Vx = d->Jdiff.bottomRows(nv);
  data->Ax.setZero();
  data->Au.setZero();
  if (data->compute_acceleration and d->joint != nullptr) {
    data->Ax = d->joint->da_dx;
    data->Au = d->joint->da_du;
  }
}

template <typename Scalar>
std::shared_ptr<TaskDataAbstractTpl<Scalar>>
TaskModelJointPositionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
TaskModelJointPositionTpl<NewScalar> TaskModelJointPositionTpl<Scalar>::cast()
    const {
  typedef TaskModelJointPositionTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  return ReturnType(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      xref_.template cast<NewScalar>(), aref_.template cast<NewScalar>(), nu_);
}

template <typename Scalar>
const typename TaskModelJointPositionTpl<Scalar>::VectorXs&
TaskModelJointPositionTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
const typename TaskModelJointPositionTpl<Scalar>::VectorXs&
TaskModelJointPositionTpl<Scalar>::get_acceleration_reference() const {
  return aref_;
}

template <typename Scalar>
void TaskModelJointPositionTpl<Scalar>::set_reference(const VectorXs& xref) {
  xref_ = xref;
}

template <typename Scalar>
void TaskModelJointPositionTpl<Scalar>::set_acceleration_reference(
    const VectorXs& aref) {
  aref_ = aref;
}

template <typename Scalar>
void TaskModelJointPositionTpl<Scalar>::print(std::ostream& os) const {
  typedef typename ScalarSelector<Scalar>::type PrintableScalar;
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "TaskModelJointPosition {xref="
     << xref_.transpose().template cast<PrintableScalar>().format(fmt)
     << ", aref="
     << aref_.transpose().template cast<PrintableScalar>().format(fmt) << "}";
}

}  // namespace crocoddyl
