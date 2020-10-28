///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/constraint.hpp"

namespace crocoddyl {

template <typename Scalar>
ConstraintModelNumDiffTpl<Scalar>::ConstraintModelNumDiffTpl(const boost::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_nu(), model->get_ng(), model->get_nh()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
ConstraintModelNumDiffTpl<Scalar>::~ConstraintModelNumDiffTpl() {}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>& x,
                                             const Eigen::Ref<const VectorXs>& u) {
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);
  data_nd->data_0->g.setZero();
  data_nd->data_0->h.setZero();
  model_->calc(data_nd->data_0, x, u);
  data_nd->g = data_nd->data_0->g;
  data_nd->h = data_nd->data_0->h;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x,
                                                 const Eigen::Ref<const VectorXs>& u) {
  boost::shared_ptr<Data> data_nd = boost::static_pointer_cast<Data>(data);

  const VectorXs& g0 = data_nd->g;
  const VectorXs& h0 = data_nd->h;
  assertStableStateFD(x);

  // Computing the d constraint(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    // x + dx
    data_nd->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, data_nd->dx, data_nd->xp);
    // call the update function on the pinocchio data
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](data_nd->xp);
    }
    // constraints(x+dx, u)
    model_->calc(data_nd->data_x[ix], data_nd->xp, u);
    // Gx, Hx
    data_nd->Gx.col(ix) = (data_nd->data_x[ix]->g - g0) / disturbance_;
    data_nd->Hx.col(ix) = (data_nd->data_x[ix]->h - h0) / disturbance_;
    data_nd->dx(ix) = 0.0;
  }

  // Computing the d constraint(x,u) / du
  data_nd->du.setZero();
  // call the update function on the pinocchio data
  for (std::size_t i = 0; i < reevals_.size(); ++i) {
    reevals_[i](x);
  }
  for (std::size_t iu = 0; iu < model_->get_nu(); ++iu) {
    // up = u + du
    data_nd->du(iu) = disturbance_;
    data_nd->up = u + data_nd->du;
    // constraint(x, u+du)
    model_->calc(data_nd->data_u[iu], x, data_nd->up);
    // Gu, Hu
    data_nd->Gu.col(iu) = (data_nd->data_u[iu]->g - g0) / disturbance_;
    data_nd->Hu.col(iu) = (data_nd->data_u[iu]->h - h0) / disturbance_;
    data_nd->du(iu) = 0.0;
  }
}

template <typename Scalar>
boost::shared_ptr<ConstraintDataAbstractTpl<Scalar> > ConstraintModelNumDiffTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const boost::shared_ptr<ConstraintModelAbstractTpl<Scalar> >& ConstraintModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar& ConstraintModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::set_disturbance(const Scalar& disturbance) {
  disturbance_ = disturbance;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::set_reevals(const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void ConstraintModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
