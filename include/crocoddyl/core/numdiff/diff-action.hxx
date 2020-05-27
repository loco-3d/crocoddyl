///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelNumDiffTpl<Scalar>::DifferentialActionModelNumDiffTpl(boost::shared_ptr<Base> model,
                                                                             bool with_gauss_approx)
    : Base(model->get_state(), model->get_nu(), model->get_nr()), model_(model) {
  with_gauss_approx_ = with_gauss_approx;
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
  if (with_gauss_approx_ && nr_ == 1) throw_pretty("No Gauss approximation possible with nr = 1");
}

template <typename Scalar>
DifferentialActionModelNumDiffTpl<Scalar>::~DifferentialActionModelNumDiffTpl() {}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>& x,
                                                     const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* data_nd = static_cast<Data*>(data.get());
  model_->calc(data_nd->data_0, x, u);
  data->cost = data_nd->data_0->cost;
  data->xout = data_nd->data_0->xout;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                         const Eigen::Ref<const VectorXs>& x,
                                                         const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);

  const VectorXs& xn0 = data_nd->data_0->xout;
  const Scalar& c0 = data_nd->data_0->cost;
  data->xout = data_nd->data_0->xout;
  data->cost = data_nd->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    data_nd->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, data_nd->dx, data_nd->xp);
    model_->calc(data_nd->data_x[ix], data_nd->xp, u);

    const VectorXs& xn = data_nd->data_x[ix]->xout;
    const Scalar& c = data_nd->data_x[ix]->cost;
    data->Fx.col(ix) = (xn - xn0) / disturbance_;

    data->Lx(ix) = (c - c0) / disturbance_;
    data_nd->Rx.col(ix) = (data_nd->data_x[ix]->r - data_nd->data_0->r) / disturbance_;
    data_nd->dx(ix) = 0.0;
  }

  // Computing the d action(x,u) / du
  data_nd->du.setZero();
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    data_nd->du(iu) = disturbance_;
    model_->calc(data_nd->data_u[iu], x, u + data_nd->du);

    const VectorXs& xn = data_nd->data_u[iu]->xout;
    const Scalar& c = data_nd->data_u[iu]->cost;
    data->Fu.col(iu) = (xn - xn0) / disturbance_;

    data->Lu(iu) = (c - c0) / disturbance_;
    data_nd->Ru.col(iu) = (data_nd->data_u[iu]->r - data_nd->data_0->r) / disturbance_;
    data_nd->du(iu) = 0.0;
  }

  if (with_gauss_approx_) {
    data->Lxx = data_nd->Rx.transpose() * data_nd->Rx;
    data->Lxu = data_nd->Rx.transpose() * data_nd->Ru;
    data->Luu = data_nd->Ru.transpose() * data_nd->Ru;
  }
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> > DifferentialActionModelNumDiffTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
const boost::shared_ptr<DifferentialActionModelAbstractTpl<Scalar> >&
DifferentialActionModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar& DifferentialActionModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::set_disturbance(const Scalar& disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

template <typename Scalar>
bool DifferentialActionModelNumDiffTpl<Scalar>::get_with_gauss_approx() {
  return with_gauss_approx_;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /** x */) {
  // TODO(cmastalli): First we need to do it AMNumDiff and then to replicate it.
}

}  // namespace crocoddyl
