///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ResidualModelTaskFirstOrderTpl<Scalar>::ResidualModelTaskFirstOrderTpl(
    std::shared_ptr<TaskModelAbstract> task,
    std::shared_ptr<GuidanceModelAbstract> guidance)
    : Base(task->get_state(), task->get_nr(), task->get_nu(),
           task->get_q_dependent(), task->get_v_dependent(),
           task->get_u_dependent()),
      task_(std::move(task)),
      guidance_(std::move(guidance)) {}

template <typename Scalar>
void ResidualModelTaskFirstOrderTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  d->task->compute_acceleration = false;
  task_->calc(d->task, x, u);
  guidance_->calc(d->guidance, d->task->y);
  d->r = d->task->v;
  d->r.noalias() -= d->guidance->g;
}

template <typename Scalar>
void ResidualModelTaskFirstOrderTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  task_->calcDiff(d->task, x, u);
  guidance_->calcDiff(d->guidance, d->task->y);
  d->Rx = d->task->Vx;
  d->Rx.noalias() -= d->guidance->Ge * d->task->Yx;
  d->Ru = d->task->Vu;
  d->Ru.noalias() -= d->guidance->Ge * d->task->Yu;
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelTaskFirstOrderTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelTaskFirstOrderTpl<NewScalar>
ResidualModelTaskFirstOrderTpl<Scalar>::cast() const {
  typedef ResidualModelTaskFirstOrderTpl<NewScalar> ReturnType;
  return ReturnType(task_->template cast<NewScalar>(),
                    guidance_->template cast<NewScalar>());
}

template <typename Scalar>
const std::shared_ptr<
    typename ResidualModelTaskFirstOrderTpl<Scalar>::TaskModelAbstract>&
ResidualModelTaskFirstOrderTpl<Scalar>::get_task() const {
  return task_;
}

template <typename Scalar>
const std::shared_ptr<
    typename ResidualModelTaskFirstOrderTpl<Scalar>::GuidanceModelAbstract>&
ResidualModelTaskFirstOrderTpl<Scalar>::get_guidance() const {
  return guidance_;
}

template <typename Scalar>
void ResidualModelTaskFirstOrderTpl<Scalar>::set_guidance(
    std::shared_ptr<GuidanceModelAbstract> guidance) {
  if (!guidance) {
    throw_pretty("Invalid argument: guidance model must not be null");
  }
  guidance_ = std::move(guidance);
  checkDimensions();
}

template <typename Scalar>
void ResidualModelTaskFirstOrderTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelTaskFirstOrder {nr=" << nr_ << ", nu=" << nu_
     << ", task=" << *task_ << ", guidance=" << *guidance_ << "}";
}

template <typename Scalar>
void ResidualModelTaskFirstOrderTpl<Scalar>::checkDimensions() const {
  if (!task_) {
    throw_pretty("Invalid argument: task must not be null");
  }
  if (!guidance_) {
    throw_pretty("Invalid argument: guidance model must not be null");
  }
  if (task_->get_nr() != guidance_->get_nr()) {
    throw_pretty("Invalid argument: task and guidance dimensions must match ("
                 << task_->get_nr() << " provided for the task, "
                 << guidance_->get_nr() << " provided for the guidance model)");
  }
  if (task_->get_nr() != nr_) {
    throw_pretty("Internal error: residual and task dimensions differ");
  }
}

}  // namespace crocoddyl
