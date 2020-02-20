///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ActivationModelNumDiffTpl<Scalar>::ActivationModelNumDiffTpl(boost::shared_ptr<Base> model)
    : Base(model->get_nr()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar>
ActivationModelNumDiffTpl<Scalar>::~ActivationModelNumDiffTpl() {}

template <typename Scalar>
void ActivationModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>& r) {
  if (static_cast<std::size_t>(r.size()) != model_->get_nr()) {
    throw_pretty("Invalid argument: "
                 << "r has wrong dimension (it should be " + std::to_string(model_->get_nr()) + ")");
  }
  boost::shared_ptr<ActivationDataNumDiffTpl<Scalar> > data_nd =
      boost::static_pointer_cast<ActivationDataNumDiffTpl<Scalar> >(data);
  model_->calc(data_nd->data_0, r);
  data->a_value = data_nd->data_0->a_value;
}

template <typename Scalar>
void ActivationModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& r) {
  if (static_cast<std::size_t>(r.size()) != model_->get_nr()) {
    throw_pretty("Invalid argument: "
                 << "r has wrong dimension (it should be " + std::to_string(model_->get_nr()) + ")");
  }
  boost::shared_ptr<ActivationDataNumDiffTpl<Scalar> > data_nd =
      boost::static_pointer_cast<ActivationDataNumDiffTpl<Scalar> >(data);

  const Scalar& a_value0 = data_nd->data_0->a_value;
  data->a_value = data_nd->data_0->a_value;
  const std::size_t& nr = model_->get_nr();

  // Computing the d activation(r) / dr
  data_nd->rp = r;
  for (unsigned int i_r = 0; i_r < nr; ++i_r) {
    data_nd->rp(i_r) += disturbance_;
    model_->calc(data_nd->data_rp[i_r], data_nd->rp);
    data_nd->rp(i_r) -= disturbance_;
    data->Ar(i_r) = (data_nd->data_rp[i_r]->a_value - a_value0) / disturbance_;
  }

  // Computing the d^2 action(r) / dr^2
  data->Arr.noalias() = data->Ar * data->Ar.transpose();
}

template <typename Scalar>
boost::shared_ptr<ActivationDataAbstractTpl<Scalar> > ActivationModelNumDiffTpl<Scalar>::createData() {
  return boost::make_shared<ActivationDataNumDiffTpl<Scalar> >(this);
}

template <typename Scalar>
const boost::shared_ptr<ActivationModelAbstractTpl<Scalar> >& ActivationModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar& ActivationModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void ActivationModelNumDiffTpl<Scalar>::set_disturbance(const Scalar& disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

}  // namespace crocoddyl
