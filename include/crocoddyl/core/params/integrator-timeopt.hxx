///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <cmath>

#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/params/integrator-timeopt.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratorTimeoptParamsTpl<Scalar>::IntegratorTimeoptParamsTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<IntegratorTime> integrator_time)
    : Base(state, 1), integrator_time_(integrator_time) {
  if (state == nullptr) {
    throw_pretty("Invalid argument: state is null");
  }
  if (integrator_time_ == nullptr) {
    throw_pretty("Invalid argument: integrator_time is null");
  }
  if (!integrator_time_->get_timeopt()) {
    throw_pretty("Invalid argument: integrator_time.timeopt should be true");
  }
  if (this->state_->get_ndx() != 2 * this->state_->get_nv()) {
    throw_pretty(
        "Invalid argument: IntegratorTimeoptParams requires a second-order "
        "state with ndx = 2 * nv");
  }
}

template <typename Scalar>
std::shared_ptr<
    typename IntegratorTimeoptParamsTpl<Scalar>::IntegratorTimeoptParamsData>
IntegratorTimeoptParamsTpl<Scalar>::castData(
    const std::shared_ptr<ParamsDataAbstract>& data) const {
  const std::shared_ptr<IntegratorTimeoptParamsData> d =
      std::dynamic_pointer_cast<IntegratorTimeoptParamsData>(data);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: data is not an IntegratorTimeoptParamsData");
  }
  return d;
}

template <typename Scalar>
void IntegratorTimeoptParamsTpl<Scalar>::update(
    const std::shared_ptr<ParamsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != this->np_) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(this->np_) + ")");
  }

  const std::shared_ptr<IntegratorTimeoptParamsData> d = castData(data);
  using std::exp;
  const Scalar dt = exp(p[0]);
  d->p = p;
  d->dt = dt;
  d->dt_dp = dt;
  integrator_time_->set_time_step(dt);
}

template <typename Scalar>
void IntegratorTimeoptParamsTpl<Scalar>::computeParamSensitivity(
    const std::shared_ptr<ActionDataAbstract>& data,
    const std::shared_ptr<ParamsDataAbstract>& params,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  typedef IntegratedActionDataEulerTpl<Scalar> IntegratedActionDataEuler;

  if (static_cast<std::size_t>(x.size()) != this->state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be " +
                 std::to_string(this->state_->get_nx()) + ")");
  }
  const std::shared_ptr<IntegratorTimeoptParamsData> d = castData(params);
  const std::shared_ptr<IntegratedActionDataEuler> action_data =
      std::dynamic_pointer_cast<IntegratedActionDataEuler>(data);
  if (action_data == nullptr) {
    throw_pretty(
        "Invalid argument: IntegratorTimeoptParams expects "
        "IntegratedActionDataEuler");
  }
  if (action_data->dynamics == nullptr) {
    throw_pretty(
        "Invalid argument: action data does not contain dynamics data");
  }

  const std::size_t nq = this->state_->get_nq();
  const std::size_t nv = this->state_->get_nv();
  const Scalar dt = integrator_time_->get_time_step();
  const Scalar dt2 = integrator_time_->get_time_step2();

  d->dx_dp.setZero();
  d->dx_dp.topRows(nv).col(0).noalias() =
      dt * x.segment(nq, nv) + Scalar(2.) * dt2 * action_data->dynamics->vdot;
  d->dx_dp.bottomRows(nv).col(0).noalias() = dt * action_data->dynamics->vdot;
  this->state_->JintegrateTransport(x, action_data->dx, d->dx_dp, first);
}

template <typename Scalar>
std::shared_ptr<typename IntegratorTimeoptParamsTpl<Scalar>::ParamsDataAbstract>
IntegratorTimeoptParamsTpl<Scalar>::createData() {
  return std::allocate_shared<IntegratorTimeoptParamsData>(
      Eigen::aligned_allocator<IntegratorTimeoptParamsData>(), this);
}

template <typename Scalar>
bool IntegratorTimeoptParamsTpl<Scalar>::checkData(
    const std::shared_ptr<ParamsDataAbstract>& data) const {
  return std::dynamic_pointer_cast<IntegratorTimeoptParamsData>(data) !=
             nullptr &&
         Base::checkData(data);
}

template <typename Scalar>
typename IntegratorTimeoptParamsTpl<Scalar>::VectorXs
IntegratorTimeoptParamsTpl<Scalar>::rand() const {
  VectorXs p(1);
  const VectorXs sample =
      vector_random_cast<Scalar, Eigen::Dynamic, Eigen::ColMajor,
                         Eigen::Dynamic>(1);
  const Scalar alpha = Scalar(0.5) * (sample[0] + Scalar(1.));
  const Scalar dt = Scalar(1e-4) + alpha * (Scalar(1e-2) - Scalar(1e-4));
  using std::log;
  p[0] = log(dt);
  return p;
}

template <typename Scalar>
const std::shared_ptr<
    typename IntegratorTimeoptParamsTpl<Scalar>::IntegratorTime>&
IntegratorTimeoptParamsTpl<Scalar>::get_integrator_time() const {
  return integrator_time_;
}

template <typename Scalar>
template <typename NewScalar>
IntegratorTimeoptParamsTpl<NewScalar> IntegratorTimeoptParamsTpl<Scalar>::cast()
    const {
  typedef IntegratorTimeoptParamsTpl<NewScalar> ReturnType;
  ReturnType ret(std::static_pointer_cast<StateAbstractTpl<NewScalar> >(
                     this->state_->template cast<NewScalar>()),
                 std::make_shared<IntegratorTimeTpl<NewScalar> >(
                     integrator_time_->template cast<NewScalar>()));
  ret.set_lb(this->lb_.template cast<NewScalar>());
  ret.set_ub(this->ub_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void IntegratorTimeoptParamsTpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratorTimeoptParams {np=" << this->np_
     << ", dt=" << integrator_time_->get_time_step() << "}";
}

}  // namespace crocoddyl
