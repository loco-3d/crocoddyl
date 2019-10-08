///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/activation.hpp"

namespace crocoddyl {

ActivationModelNumDiff::ActivationModelNumDiff(ActivationModelAbstract& model)
    : ActivationModelAbstract(model.get_nr()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
}

ActivationModelNumDiff::~ActivationModelNumDiff() {}

void ActivationModelNumDiff::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>& r) {
  assert(r.size() == model_.get_nr() && "r has wrong dimension");
  boost::shared_ptr<ActivationDataNumDiff> data_nd = boost::static_pointer_cast<ActivationDataNumDiff>(data);
  model_.calc(data_nd->data_0, r);
  data->a_value = data_nd->data_0->a_value;
}

void ActivationModelNumDiff::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  assert(r.size() == model_.get_nr() && "r has wrong dimension");
  boost::shared_ptr<ActivationDataNumDiff> data_nd = boost::static_pointer_cast<ActivationDataNumDiff>(data);

  if (recalc) {
    model_.calc(data_nd->data_0, r);
  }
  const double& a_value0 = data_nd->data_0->a_value;
  data->a_value = data_nd->data_0->a_value;
  const unsigned int& nr = model_.get_nr();

  // Computing the d activation(r) / dr
  data_nd->rp = r;
  for (unsigned int i_r = 0; i_r < nr; ++i_r) {
    data_nd->rp(i_r) += disturbance_;
    model_.calc(data_nd->data_rp[i_r], data_nd->rp);
    data_nd->rp(i_r) -= disturbance_;
    data->Ar(i_r) = (data_nd->data_rp[i_r]->a_value - a_value0) / disturbance_;
  }

  // Computing the d^2 action(r) / dr^2
  data->Arr.noalias() = data->Ar * data->Ar.transpose();
}

ActivationModelAbstract& ActivationModelNumDiff::get_model() const { return model_; }

const double& ActivationModelNumDiff::get_disturbance() const { return disturbance_; }

boost::shared_ptr<ActivationDataAbstract> ActivationModelNumDiff::createData() {
  return boost::make_shared<ActivationDataNumDiff>(this);
}

}  // namespace crocoddyl
