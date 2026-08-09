///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ResidualModelTaskSecondOrderTpl<Scalar>::ResidualModelTaskSecondOrderTpl(
    std::shared_ptr<TaskModelAbstract> task,
    std::shared_ptr<GuidanceModelAbstract> guidance, const MatrixXs& gain)
    : Base(task->get_state(), task->get_nr(), task->get_nu(),
           task->get_q_dependent(), task->get_v_dependent(),
           task->get_u_dependent()),
      task_(std::move(task)),
      guidance_(std::move(guidance)),
      gain_(gain) {}

template <typename Scalar>
ResidualModelTaskSecondOrderTpl<Scalar>::ResidualModelTaskSecondOrderTpl(
    std::shared_ptr<TaskModelAbstract> task,
    std::shared_ptr<GuidanceModelAbstract> guidance,
    const VectorXs& diagonal_gain)
    : ResidualModelTaskSecondOrderTpl(std::move(task), std::move(guidance),
                                      MatrixXs(diagonal_gain.asDiagonal())) {}

template <typename Scalar>
ResidualModelTaskSecondOrderTpl<Scalar>::ResidualModelTaskSecondOrderTpl(
    std::shared_ptr<TaskModelAbstract> task,
    std::shared_ptr<GuidanceModelAbstract> guidance, const Scalar& gain)
    : Base(task->get_state(), task->get_nr(), task->get_nu(),
           task->get_q_dependent(), task->get_v_dependent(),
           task->get_u_dependent()),
      task_(std::move(task)),
      guidance_(std::move(guidance)),
      gain_(MatrixXs::Identity(nr_, nr_) * gain) {}

template <typename Scalar>
void ResidualModelTaskSecondOrderTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  d->task->compute_acceleration = true;
  task_->calc(d->task, x, u);
  guidance_->calc(d->guidance, d->task->y);
  d->v_error = d->task->v;
  d->v_error.noalias() -= d->guidance->g;
  d->r = d->task->a;
  d->r.noalias() += gain_ * d->v_error;
}

template <typename Scalar>
void ResidualModelTaskSecondOrderTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  task_->calcDiff(d->task, x, u);
  guidance_->calcDiff(d->guidance, d->task->y);
  d->v_error_x = d->task->Vx;
  d->v_error_x.noalias() -= d->guidance->Ge * d->task->Yx;
  d->Rx = d->task->Ax;
  d->Rx.noalias() += gain_ * d->v_error_x;
  d->v_error_u = d->task->Vu;
  d->v_error_u.noalias() -= d->guidance->Ge * d->task->Yu;
  d->Ru = d->task->Au;
  d->Ru.noalias() += gain_ * d->v_error_u;
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelTaskSecondOrderTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelTaskSecondOrderTpl<NewScalar>
ResidualModelTaskSecondOrderTpl<Scalar>::cast() const {
  typedef ResidualModelTaskSecondOrderTpl<NewScalar> ReturnType;
  typename ReturnType::MatrixXs gain = gain_.template cast<NewScalar>();
  return ReturnType(task_->template cast<NewScalar>(),
                    guidance_->template cast<NewScalar>(), gain);
}

template <typename Scalar>
const std::shared_ptr<
    typename ResidualModelTaskSecondOrderTpl<Scalar>::TaskModelAbstract>&
ResidualModelTaskSecondOrderTpl<Scalar>::get_task() const {
  return task_;
}

template <typename Scalar>
const std::shared_ptr<
    typename ResidualModelTaskSecondOrderTpl<Scalar>::GuidanceModelAbstract>&
ResidualModelTaskSecondOrderTpl<Scalar>::get_guidance() const {
  return guidance_;
}

template <typename Scalar>
const typename ResidualModelTaskSecondOrderTpl<Scalar>::MatrixXs&
ResidualModelTaskSecondOrderTpl<Scalar>::get_gain() const {
  return gain_;
}

template <typename Scalar>
void ResidualModelTaskSecondOrderTpl<Scalar>::set_guidance(
    std::shared_ptr<GuidanceModelAbstract> guidance) {
  if (!guidance) {
    throw_pretty("Invalid argument: guidance model must not be null");
  }
  guidance_ = std::move(guidance);
  checkDimensions();
}

template <typename Scalar>
void ResidualModelTaskSecondOrderTpl<Scalar>::set_gain(const MatrixXs& gain) {
  if (static_cast<std::size_t>(gain.rows()) != nr_ ||
      static_cast<std::size_t>(gain.cols()) != nr_) {
    throw_pretty("Invalid argument: task gain has wrong dimension ("
                 << gain.rows() << "x" << gain.cols() << " provided, expected "
                 << nr_ << "x" << nr_ << ")");
  }
  gain_ = gain;
}

template <typename Scalar>
void ResidualModelTaskSecondOrderTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelTaskSecondOrder {nr=" << nr_ << ", nu=" << nu_
     << ", task=" << *task_ << ", guidance=" << *guidance_
     << ", gain_dim=" << gain_.rows() << "x" << gain_.cols() << "}";
}

template <typename Scalar>
void ResidualModelTaskSecondOrderTpl<Scalar>::checkDimensions() const {
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
  if (static_cast<std::size_t>(gain_.rows()) != nr_ ||
      static_cast<std::size_t>(gain_.cols()) != nr_) {
    throw_pretty("Internal error: task gain has wrong dimension");
  }
}

}  // namespace crocoddyl
