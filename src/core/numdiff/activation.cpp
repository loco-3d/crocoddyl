///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/activation.hpp"

namespace crocoddyl {

ActivationModelNumDiff::ActivationModelNumDiff(boost::shared_ptr<ActivationModelAbstract> model)
    : ActivationModelAbstract(model->get_nr()), model_(model) {
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
}

ActivationModelNumDiff::~ActivationModelNumDiff() {}

void ActivationModelNumDiff::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>& r) {
  if (static_cast<std::size_t>(r.size()) != model_->get_nr()) {
    throw CrocoddylException("r has wrong dimension (it should be " + std::to_string(model_->get_nr()) + ")");
  }
  boost::shared_ptr<ActivationDataNumDiff> data_nd = boost::static_pointer_cast<ActivationDataNumDiff>(data);
  model_->calc(data_nd->data_0, r);
  data->a_value = data_nd->data_0->a_value;
}

void ActivationModelNumDiff::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  if (static_cast<std::size_t>(r.size()) != model_->get_nr()) {
    throw CrocoddylException("r has wrong dimension (it should be " + std::to_string(model_->get_nr()) + ")");
  }
  boost::shared_ptr<ActivationDataNumDiff> data_nd = boost::static_pointer_cast<ActivationDataNumDiff>(data);

  if (recalc) {
    model_->calc(data_nd->data_0, r);
  }
  const double& a_value0 = data_nd->data_0->a_value;
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

boost::shared_ptr<ActivationDataAbstract> ActivationModelNumDiff::createData() {
  return boost::make_shared<ActivationDataNumDiff>(this);
}

const boost::shared_ptr<ActivationModelAbstract>& ActivationModelNumDiff::get_model() const { return model_; }

const double& ActivationModelNumDiff::get_disturbance() const { return disturbance_; }

void ActivationModelNumDiff::set_disturbance(const double& disturbance) {
  if (disturbance < 0.) {
    throw CrocoddylException("Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

}  // namespace crocoddyl
