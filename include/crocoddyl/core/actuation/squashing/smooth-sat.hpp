///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_
#define CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_

#include "crocoddyl/core/actuation/squashing-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class SquashingModelSmoothSatTpl : public SquashingModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef SquashingModelAbstractTpl<Scalar> Base;
  typedef SquashingDataAbstractTpl<Scalar> SquashingDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  SquashingModelSmoothSatTpl(const Eigen::Ref<const VectorXs>& s_lb, const Eigen::Ref<const VectorXs>& s_ub,
                             const std::size_t& ns)
      : Base(ns) {
    s_lb_ = s_lb;
    s_ub_ = s_ub;

    u_lb_ = s_lb_;
    u_ub_ = s_ub_;

    smooth_ = 0.1;

    d_ = (s_ub_ - s_lb_) * smooth_;
    a_ = d_.array() * d_.array();
  }

  ~SquashingModelSmoothSatTpl();

  void calc(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const VectorXs>& u) {
    // Squashing function used: "Smooth abs":
    // s(u) = 0.5*(lb + ub + sqrt(smooth + (u - lb)^2) - sqrt(smooth + (u - ub)^2))
    data->s = 0.5 * (Eigen::sqrt(Eigen::pow((u - s_lb_).array(), 2) + a_.array()) -
                     Eigen::sqrt(Eigen::pow((u - s_ub_).array(), 2) + a_.array()) + s_lb_.array() + s_ub_.array());
  }

  void calcDiff(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const VectorXs>& u) {
    data->ds_du.diagonal() =
        0.5 * (Eigen::pow(a_.array() + Eigen::pow((u - s_lb_).array(), 2), -0.5).array() * (u - s_lb_).array() -
               Eigen::pow(a_.array() + Eigen::pow((u - s_ub_).array(), 2), -0.5).array() * (u - s_ub_).array());
  }

  const Scalar& get_smooth() const { return smooth_; };
  void set_smooth(const Scalar& smooth) {
    if (smooth < 0.) {
      throw_pretty("Invalid argument: "
                   << "Smooth value has to be positive");
    }
    smooth_ = smooth;

    d_ = (s_ub_ - s_lb_) * smooth_;
    a_ = d_.array() * d_.array();

    u_lb_ = s_lb_;
    u_ub_ = s_ub_;
  }

  const VectorXs& get_d() const { return d_; };

 private:
  VectorXs a_;
  VectorXs d_;

  Scalar smooth_;

 protected:
  using Base::s_lb_;
  using Base::s_ub_;

  using Base::u_lb_;
  using Base::u_ub_;
};
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_
