///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {
template <typename Scalar>
ActionModelUnicycleTpl<Scalar>::ActionModelUnicycleTpl()
    : ActionModelAbstractTpl<Scalar>(
          std::make_shared<StateVectorTpl<Scalar> >(3), 2, 5),
      dt_(Scalar(0.1)) {
  cost_weights_ << Scalar(10.), Scalar(1.);
}

template <typename Scalar>
void ActionModelUnicycleTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstractTpl<Scalar> >& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const Scalar c = cos(x[2]);
  const Scalar s = sin(x[2]);
  d->xnext << x[0] + c * u[0] * dt_, x[1] + s * u[0] * dt_, x[2] + u[1] * dt_;
  d->r.template head<3>() = cost_weights_[0] * x;
  d->r.template tail<2>() = cost_weights_[1] * u;
  d->cost = Scalar(0.5) * d->r.dot(d->r);
}

template <typename Scalar>
void ActionModelUnicycleTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstractTpl<Scalar> >& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  d->xnext = x;
  d->r.template head<3>() = cost_weights_[0] * x;
  d->r.template tail<2>().setZero();
  d->cost = Scalar(0.5) * d->r.template head<3>().dot(d->r.template head<3>());
}

template <typename Scalar>
void ActionModelUnicycleTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstractTpl<Scalar> >& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const Scalar c = static_cast<Scalar>(cos(x[2]));
  const Scalar s = static_cast<Scalar>(sin(x[2]));
  const Scalar w_x = cost_weights_[0] * cost_weights_[0];
  const Scalar w_u = cost_weights_[1] * cost_weights_[1];
  d->Lx = x * w_x;
  d->Lu = u * w_u;
  d->Lxx.diagonal().setConstant(w_x);
  d->Luu.diagonal().setConstant(w_u);
  d->Fx(0, 2) = -s * u[0] * dt_;
  d->Fx(1, 2) = c * u[0] * dt_;
  d->Fu(0, 0) = c * dt_;
  d->Fu(1, 0) = s * dt_;
  d->Fu(2, 1) = dt_;
}

template <typename Scalar>
void ActionModelUnicycleTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstractTpl<Scalar> >& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const Scalar w_x = cost_weights_[0] * cost_weights_[0];
  d->Lx = x * w_x;
  d->Lxx.diagonal().setConstant(w_x);
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
ActionModelUnicycleTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
template <typename NewScalar>
ActionModelUnicycleTpl<NewScalar> ActionModelUnicycleTpl<Scalar>::cast() const {
  typedef ActionModelUnicycleTpl<NewScalar> ReturnType;
  ReturnType ret;
  return ret;
}

template <typename Scalar>
bool ActionModelUnicycleTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void ActionModelUnicycleTpl<Scalar>::print(std::ostream& os) const {
  os << "ActionModelUnicycle {dt=" << dt_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s&
ActionModelUnicycleTpl<Scalar>::get_cost_weights() const {
  return cost_weights_;
}

template <typename Scalar>
void ActionModelUnicycleTpl<Scalar>::set_cost_weights(
    const typename MathBase::Vector2s& weights) {
  cost_weights_ = weights;
}

template <typename Scalar>
Scalar ActionModelUnicycleTpl<Scalar>::get_dt() const {
  return dt_;
}

template <typename Scalar>
void ActionModelUnicycleTpl<Scalar>::set_dt(const Scalar dt) {
  if (dt <= 0)
    throw_pretty("Invalid argument: dt should be strictly positive.");
  dt_ = dt;
}

}  // namespace crocoddyl
