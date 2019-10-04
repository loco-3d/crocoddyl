///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/smooth-abs.hpp"
#include <iostream>
namespace crocoddyl {

ActivationModelSmoothAbs::ActivationModelSmoothAbs(unsigned int const& nr) : ActivationModelAbstract(nr) {}

ActivationModelSmoothAbs::~ActivationModelSmoothAbs() {}

void ActivationModelSmoothAbs::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                    const Eigen::Ref<const Eigen::VectorXd>& r) {
  assert(r.size() == nr_ && "r has wrong dimension");
  boost::shared_ptr<ActivationDataSmoothAbs> d = boost::static_pointer_cast<ActivationDataSmoothAbs>(data);

  d->a = (r.array().cwiseAbs2().array() + 1).array().cwiseSqrt();
  data->a_value = d->a.squaredNorm();
}

void ActivationModelSmoothAbs::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                        const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  assert(r.size() == nr_ && "r has wrong dimension");
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
