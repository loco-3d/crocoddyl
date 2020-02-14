///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include <stdexcept>

namespace crocoddyl {

class ActivationModelQuad : public ActivationModelAbstract {
 public:
  explicit ActivationModelQuad(const std::size_t& nr);
  ~ActivationModelQuad();

  void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  boost::shared_ptr<ActivationDataAbstract> createData();
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
