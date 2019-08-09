///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_

#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

class ActivationModelWeightedQuad : public ActivationModelAbstract {
 public:
  explicit ActivationModelWeightedQuad(const Eigen::VectorXd& weights);
  ~ActivationModelWeightedQuad();

  void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r,
                const bool& recalc = true);
  boost::shared_ptr<ActivationDataAbstract> createData();

 private:
  Eigen::VectorXd weights_;
};

struct ActivationDataWeightedQuad : public ActivationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Activation>
  explicit ActivationDataWeightedQuad(Activation* const activation)
      : ActivationDataAbstract(activation), Wr(Eigen::VectorXd::Zero(activation->get_nr())) {}

  Eigen::VectorXd Wr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
