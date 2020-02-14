///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_BARRIER_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_BARRIER_HPP_

#include <stdexcept>
#include <math.h>
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

struct ActivationBounds {
  ActivationBounds(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper, const double& b = 1.)
      : lb(lower), ub(upper), beta(b) {
    if (lb.size() != ub.size()) {
      throw_pretty("Invalid argument: "
                   << "The lower and upper bounds don't have the same dimension (lb,ub dimensions equal to " +
                          std::to_string(lb.size()) + "," + std::to_string(ub.size()) + ", respectively)");
    }
    if (beta < 0 || beta > 1.) {
      throw_pretty("Invalid argument: "
                   << "The range of beta is between 0 and 1");
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(lb.size()); ++i) {
      if (std::isfinite(lb(i)) && std::isfinite(ub(i))) {
        if (lb(i) - ub(i) > 0) {
          throw_pretty("Invalid argument: "
                       << "The lower and upper bounds are badly defined; ub has to be bigger / equals to lb");
        }
      }
    }

    if (beta >= 0 && beta <= 1.) {
      Eigen::VectorXd m = 0.5 * (lower + upper);
      Eigen::VectorXd d = 0.5 * (upper - lower);
      lb = m - beta * d;
      ub = m + beta * d;
    } else {
      beta = 1.;
    }
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
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
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
