///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_

#include "crocoddyl/core/activations/quadratic-barrier.hpp"

namespace crocoddyl {

class ActivationModelWeightedQuadraticBarrier : public ActivationModelAbstract {
 public:
  explicit ActivationModelWeightedQuadraticBarrier(const ActivationBounds& bounds, const Eigen::VectorXd& weights);
  ~ActivationModelWeightedQuadraticBarrier();

  void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  boost::shared_ptr<ActivationDataAbstract> createData();

  const ActivationBounds& get_bounds() const;
  const Eigen::VectorXd& get_weights() const;
  void set_bounds(const ActivationBounds& bounds);
  void set_weights(const Eigen::VectorXd& weights);

 private:
  ActivationBounds bounds_;
  Eigen::VectorXd weights_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_
