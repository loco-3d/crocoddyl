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

  SquashingModelSmoothSatTpl(const Eigen::Ref<const VectorXs>& u_lb, const Eigen::Ref<const VectorXs>& u_ub,
                             const std::size_t& ns)
      : Base(ns) {
    u_lb_ = u_lb;
    u_ub_ = u_ub;

    s_lb_ = u_lb_;
    s_ub_ = u_ub_;

    smooth_ = 0.1;

    d_ = (u_ub_ - u_lb_) * smooth_;
    a_ = d_.array() * d_.array();
  }

  ~SquashingModelSmoothSatTpl();

  void calc(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const VectorXs>& s) {
    // Squashing function used: "Smooth abs":
    // s(u) = 0.5*(lb + ub + sqrt(smooth + (u - lb)^2) - sqrt(smooth + (u - ub)^2))
    data->s = 0.5 * (Eigen::sqrt(Eigen::pow((s - u_lb_).array(), 2) + a_.array()) -
                     Eigen::sqrt(Eigen::pow((s - u_ub_).array(), 2) + a_.array()) + u_lb_.array() + u_ub_.array());
  }

  void calcDiff(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const VectorXs>& s) {
    data->du_ds.diagonal() =
        0.5 * (Eigen::pow(a_.array() + Eigen::pow((s - u_lb_).array(), 2), -0.5).array() * (s - u_lb_).array() -
               Eigen::pow(a_.array() + Eigen::pow((s - u_ub_).array(), 2), -0.5).array() * (s - u_ub_).array());
  }

  const Scalar& get_smooth() const { return smooth_; };
  void set_smooth(const Scalar& smooth) {
    if (smooth < 0.) {
      throw_pretty("Invalid argument: "
                   << "Smooth value has to be positive");
    }
    smooth_ = smooth;

    d_ = (u_ub_ - u_lb_) * smooth_;
    a_ = d_.array() * d_.array();

    s_lb_ = u_lb_;
    s_ub_ = u_ub_;
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
