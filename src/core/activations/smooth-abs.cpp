///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/activations/smooth-abs.hpp"

namespace crocoddyl {

ActivationModelSmoothAbs::ActivationModelSmoothAbs(const std::size_t& nr) : ActivationModelAbstract(nr) {}

ActivationModelSmoothAbs::~ActivationModelSmoothAbs() {}

void ActivationModelSmoothAbs::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                    const Eigen::Ref<const Eigen::VectorXd>& r) {
  if (static_cast<std::size_t>(r.size()) != nr_) {
    throw_pretty("Invalid argument: " << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
  }
  boost::shared_ptr<ActivationDataSmoothAbs> d = boost::static_pointer_cast<ActivationDataSmoothAbs>(data);

  d->a = (r.array().cwiseAbs2().array() + 1).array().cwiseSqrt();
  data->a_value = d->a.sum();
}

void ActivationModelSmoothAbs::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                        const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  if (static_cast<std::size_t>(r.size()) != nr_) {
    throw_pretty("Invalid argument: " << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
  }
  if (recalc) {
    calc(data, r);
  }

  boost::shared_ptr<ActivationDataSmoothAbs> d = boost::static_pointer_cast<ActivationDataSmoothAbs>(data);
  data->Ar = r.cwiseProduct(d->a.cwiseInverse());
  data->Arr.diagonal() = d->a.cwiseProduct(d->a).cwiseProduct(d->a).cwiseInverse();
}

boost::shared_ptr<ActivationDataAbstract> ActivationModelSmoothAbs::createData() {
  return boost::make_shared<ActivationDataSmoothAbs>(this);
}

}  // namespace crocoddyl
