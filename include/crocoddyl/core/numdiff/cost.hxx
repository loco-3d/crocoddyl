///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/cost.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelNumDiffTpl<Scalar>::CostModelNumDiffTpl(const boost::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_activation(), model->get_nu()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
CostModelNumDiffTpl<Scalar>::~CostModelNumDiffTpl() {}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                       const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);
  data_nd->data_0->cost = 0.0;
  model_->calc(data_nd->data_0, x, u);
  data_nd->cost = data_nd->data_0->cost;
  data_nd->r = data_nd->data_0->r;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);

  Scalar c0 = data_nd->cost;
  const VectorXs& r0 = data_nd->r;
  if (get_with_gauss_approx()) {
    model_->get_activation()->calc(data_nd->data_0->activation, r0);
    model_->get_activation()->calcDiff(data_nd->data_0->activation, r0);
  }
  assertStableStateFD(x);

  // Computing the d cost(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    // x + dx
    data_nd->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, data_nd->dx, data_nd->xp);
    // call the update function on the pinocchio data
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](data_nd->xp);
    }
    // cost(x+dx, u)
    model_->calc(data_nd->data_x[ix], data_nd->xp, u);
    // Lx
    data_nd->Lx(ix) = (data_nd->data_x[ix]->cost - c0) / disturbance_;
    // Check if we need to/can compute the Gauss approximation of the Hessian.
    if (get_with_gauss_approx()) {
      data_nd->Rx.col(ix) = (data_nd->data_x[ix]->r - r0) / disturbance_;
    }
    data_nd->dx(ix) = 0.0;
  }

  // Computing the d cost(x,u) / du
  data_nd->du.setZero();
  // call the update function on the pinocchio data
  for (std::size_t i = 0; i < reevals_.size(); ++i) {
    reevals_[i](x);
  }
  for (std::size_t iu = 0; iu < model_->get_nu(); ++iu) {
    // up = u + du
    data_nd->du(iu) = disturbance_;
    data_nd->up = u + data_nd->du;
    // cost(x, u+du)
    model_->calc(data_nd->data_u[iu], x, data_nd->up);
    // Lu
    data_nd->Lu(iu) = (data_nd->data_u[iu]->cost - c0) / disturbance_;
    // Check if we need to/can compute the Gauss approximation of the Hessian.
    if (get_with_gauss_approx()) {
      data_nd->Ru.col(iu) = (data_nd->data_u[iu]->r - r0) / disturbance_;
    }
    data_nd->du(iu) = 0.0;
  }

  if (get_with_gauss_approx()) {
    const MatrixXs& Arr = data_nd->data_0->activation->Arr;
    data_nd->Lxx = data_nd->Rx.transpose() * Arr * data_nd->Rx;
    data_nd->Lxu = data_nd->Rx.transpose() * Arr * data_nd->Ru;
    data_nd->Luu = data_nd->Ru.transpose() * Arr * data_nd->Ru;
  } else {
    data_nd->Lxx.fill(0.0);
    data_nd->Lxu.fill(0.0);
    data_nd->Luu.fill(0.0);
  }
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelNumDiffTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const boost::shared_ptr<CostModelAbstractTpl<Scalar> >& CostModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
Scalar CostModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::set_disturbance(Scalar disturbance) {
  disturbance_ = disturbance;
}

template <typename Scalar>
bool CostModelNumDiffTpl<Scalar>::get_with_gauss_approx() {
  return activation_->get_nr() > 0;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::set_reevals(const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void CostModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
