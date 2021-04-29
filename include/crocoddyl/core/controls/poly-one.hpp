///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_ONE_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_ONE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree one, that is a linear function
 * 
 * The size of the parameters p is twice the size of the control input u.
 * The first half of p represents the value of u at time 0.
 * The second half of p represents the value of u at time 0.5.
 */
template <typename _Scalar>
class ControlPolyOneTpl : public ControlAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ControlPolyOneTpl(const std::size_t nu);
  virtual ~ControlPolyOneTpl();

  virtual void resize(const std::size_t nu);

  virtual void value(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<VectorXs> u_out) const;

  virtual void dValue(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<MatrixXs> J_out) const;

  virtual void multiplyByDValue(double t, const Eigen::Ref<const VectorXs>& p, 
        const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const;

 protected:
  using ControlAbstractTpl<Scalar>::nu_;
  using ControlAbstractTpl<Scalar>::np_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-one.hxx"

#endif  // CROCODDYL_CORE_CONTROLS_POLY_ONE_HPP_
