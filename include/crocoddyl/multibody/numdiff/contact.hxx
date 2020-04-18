///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/numdiff/contact.hpp"

namespace crocoddyl {

template <typename Scalar>
ContactModelNumDiffTpl<Scalar>::ContactModelNumDiffTpl(const boost::shared_ptr<Base>& model)
    : Base(model->get_state(), model->get_nc(), model->get_nu()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
ContactModelNumDiffTpl<Scalar>::~ContactModelNumDiffTpl() {}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                          const Eigen::Ref<const VectorXs>& x) {
  boost::shared_ptr<ContactDataNumDiffTpl<Scalar> > data_nd =
      boost::static_pointer_cast<ContactDataNumDiffTpl<Scalar> >(data);
  model_->calc(data_nd->data_0, x);
  data_nd->a0 = data_nd->data_0->a0;
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x) {
  boost::shared_ptr<ContactDataNumDiffTpl<Scalar> > data_nd =
      boost::static_pointer_cast<ContactDataNumDiffTpl<Scalar> >(data);

  const VectorXs& a0 = data_nd->a0;

  assertStableStateFD(x);

  // Computing the d contact(x,u) / dx
  data_nd->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    // x + dx
    data_nd->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, data_nd->dx, data_nd->xp);
    // call the update function on the pinocchio data
    for (size_t i = 0; i < reevals_.size(); ++i) {
      reevals_[i](data_nd->xp);
    }
    // contact(x+dx, u)
    model_->calc(data_nd->data_x[ix], data_nd->xp);
    data_nd->da0_dx.col(ix) = (data_nd->data_x[ix]->a0 - a0) / disturbance_;
    data_nd->dx(ix) = 0.0;
  }
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                                                 const VectorXs& force) {
  if (static_cast<std::size_t>(force.size()) != model_->get_nc()) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be " << model_->get_nc() << ")");
  }

  boost::shared_ptr<ContactDataNumDiffTpl<Scalar> > data_nd =
      boost::static_pointer_cast<ContactDataNumDiffTpl<Scalar> >(data);

  model_->updateForce(data_nd->data_0, force);
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModelNumDiffTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::make_shared<ContactDataNumDiffTpl<Scalar> >(this, data);
}

template <typename Scalar>
const boost::shared_ptr<ContactModelAbstractTpl<Scalar> >& ContactModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar& ContactModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::set_disturbance(const Scalar& disturbance) {
  disturbance_ = disturbance;
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::set_reevals(const std::vector<ReevaluationFunction>& reevals) {
  reevals_ = reevals;
}

template <typename Scalar>
void ContactModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/) {
  // do nothing in the general case
}

}  // namespace crocoddyl
