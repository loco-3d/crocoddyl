///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_BARRIER_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_BARRIER_HPP_

#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

struct ActivationBounds {
  ActivationBounds(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper, const double& b = 1.)
      : lb(lower), ub(upper), beta(b) {
    assert(lb.size() == ub.size() && "The lower and upper bounds don't have the same dimension.");
    assert((beta >= 0 && beta <= 1.) && "The range of beta is between 0 and 1.");

    if (beta >= 0 && beta <= 1.) {
      Eigen::VectorXd m = 0.5 * (lower + upper);
      Eigen::VectorXd d = 0.5 * (upper - lower);
      lb = m - beta * d;
      ub = m + beta * d;
    } else {
      beta = 1.;
    }
    assert(((lb - ub).array() <= 0).all() && "The lower and upper bounds are badly defined");
  }
  ActivationBounds(const ActivationBounds& bounds) : lb(bounds.lb), ub(bounds.ub), beta(bounds.beta) {}
  ActivationBounds() : beta(1.) {}

  Eigen::VectorXd lb;
  Eigen::VectorXd ub;
  double beta;
};

class ActivationModelQuadraticBarrier : public ActivationModelAbstract {
 public:
  explicit ActivationModelQuadraticBarrier(const ActivationBounds& bounds);
  ~ActivationModelQuadraticBarrier();

  void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r,
                const bool& recalc = true);
  boost::shared_ptr<ActivationDataAbstract> createData();

  const ActivationBounds& get_bounds() const;
  void set_bounds(const ActivationBounds& bounds);

 private:
  ActivationBounds bounds_;
};

struct ActivationDataQuadraticBarrier : public ActivationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Activation>
  explicit ActivationDataQuadraticBarrier(Activation* const activation)
      : ActivationDataAbstract(activation), rlb_min_(activation->get_nr()), rub_max_(activation->get_nr()) {
    rlb_min_.fill(0);
    rub_max_.fill(0);
  }

  Eigen::ArrayXd rlb_min_;
  Eigen::ArrayXd rub_max_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_BARRIER_HPP_
