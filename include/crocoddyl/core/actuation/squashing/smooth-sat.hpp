///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, IRI: CSIC-UPC,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_
#define CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_

#include "crocoddyl/core/actuation/squashing-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class SquashingModelSmoothSatTpl : public SquashingModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(SquashingModelBase, SquashingModelSmoothSatTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef SquashingModelAbstractTpl<Scalar> Base;
  typedef SquashingDataAbstractTpl<Scalar> SquashingDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  SquashingModelSmoothSatTpl(const Eigen::Ref<const VectorXs>& u_lb,
                             const Eigen::Ref<const VectorXs>& u_ub,
                             const std::size_t ns)
      : Base(ns) {
    u_lb_ = u_lb;
    u_ub_ = u_ub;

    s_lb_ = u_lb_;
    s_ub_ = u_ub_;

    smooth_ = Scalar(0.1);

    d_ = (u_ub_ - u_lb_) * smooth_;
    a_ = d_.array() * d_.array();
  }

  virtual ~SquashingModelSmoothSatTpl() = default;

  virtual void calc(const std::shared_ptr<SquashingDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& s) override {
    // Squashing function used: "Smooth abs":
    // s(u) = 0.5*(lb + ub + sqrt(smooth + (u - lb)^2) - sqrt(smooth + (u -
    // ub)^2))
    data->u = Scalar(0.5) *
              (Eigen::sqrt(Eigen::pow((s - u_lb_).array(), 2) + a_.array()) -
               Eigen::sqrt(Eigen::pow((s - u_ub_).array(), 2) + a_.array()) +
               u_lb_.array() + u_ub_.array());
  }

  virtual void calcDiff(const std::shared_ptr<SquashingDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& s) override {
    data->du_ds.diagonal() =
        Scalar(0.5) *
        (Eigen::pow(a_.array() + Eigen::pow((s - u_lb_).array(), 2),
                    Scalar(-0.5))
                 .array() *
             (s - u_lb_).array() -
         Eigen::pow(a_.array() + Eigen::pow((s - u_ub_).array(), 2),
                    Scalar(-0.5))
                 .array() *
             (s - u_ub_).array());
  }

  template <typename NewScalar>
  SquashingModelSmoothSatTpl<NewScalar> cast() const {
    typedef SquashingModelSmoothSatTpl<NewScalar> ReturnType;
    ReturnType ret(u_lb_.template cast<NewScalar>(),
                   u_ub_.template cast<NewScalar>(), ns_);
    return ret;
  }

  const Scalar get_smooth() const { return smooth_; };
  void set_smooth(const Scalar smooth) {
    if (smooth < 0.) {
      throw_pretty("Invalid argument: " << "Smooth value has to be positive");
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

  using Base::ns_;
  using Base::s_lb_;
  using Base::s_ub_;
  using Base::u_lb_;
  using Base::u_ub_;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SquashingModelSmoothSatTpl)

#endif  // CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_
