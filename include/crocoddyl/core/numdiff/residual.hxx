///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/residual.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelNumDiffTpl<Scalar>::ResidualModelNumDiffTpl(const boost::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_nr(), model->get_nu()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
ResidualModelNumDiffTpl<Scalar>::~ResidualModelNumDiffTpl() {}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);
  model_->calc(data_nd->data_0, x, u);
  data_nd->r = data_nd->data_0->r;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>& x,
                                               const Eigen::Ref<const VectorXs>& u) {
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);

  const VectorXs& r0 = data_nd->r;
  assertStableStateFD(x);

  // Computing the d residual(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    data_nd->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, data_nd->dx, data_nd->xp);
    // call the update function
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](data_nd->xp, u);
    }
    // residual(x+dx, u)
    model_->calc(data_nd->data_x[ix], data_nd->xp, u);
    // Rx
    data_nd->Rx.col(ix) = (data_nd->data_x[ix]->r - r0) / disturbance_;
    data_nd->dx(ix) = 0.0;
  }

  // Computing the d residual(x,u) / du
  data_nd->du.setZero();
  for (std::size_t iu = 0; iu < model_->get_nu(); ++iu) {
    // up = u + du
    data_nd->du(iu) = disturbance_;
    data_nd->up = u + data_nd->du;
    // call the update function
    for (std::size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](x, data_nd->up);
    }
    // residual(x, u+du)
    model_->calc(data_nd->data_u[iu], x, data_nd->up);
    // Ru
    data_nd->Ru.col(iu) = (data_nd->data_u[iu]->r - r0) / disturbance_;
    data_nd->du(iu) = 0.0;
  }
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelNumDiffTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const boost::shared_ptr<ResidualModelAbstractTpl<Scalar> >& ResidualModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ResidualModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  disturbance_ = disturbance;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::set_reevals(const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void ResidualModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
