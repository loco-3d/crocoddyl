///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_TWO_RK3_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_TWO_RK3_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree two, that is a quadratic function
 *
 * The size of the parameters p is 3 times the size of the control input u.
 * The first third of p represents the value of u at time 0.
 * The second third of p represents the value of u at time 1/3.
 * The last third of p represents the value of u at time 2/3.
 * This parametrization is suitable to be used with the RK-3 integration scheme,
 * because it requires the value of u exactly at 0, 1/3 and 2/3.
 */
template <typename _Scalar>
class ControlParametrizationPolyTwoRK3Tpl : public ControlParametrizationAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ControlParametrizationPolyTwoRK3Tpl(const std::size_t nu);
  virtual ~ControlParametrizationPolyTwoRK3Tpl();

  virtual void resize(const std::size_t nu);

  virtual void value(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<VectorXs> u_out) const;

  virtual void value_inv(double t, const Eigen::Ref<const VectorXs>& u, Eigen::Ref<VectorXs> p_out) const;

  virtual void convert_bounds(const Eigen::Ref<const VectorXs>& u_lb, const Eigen::Ref<const VectorXs>& u_ub,
                              Eigen::Ref<VectorXs> p_lb, Eigen::Ref<VectorXs> p_ub) const;

  virtual void dValue(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<MatrixXs> J_out) const;

  virtual void multiplyByDValue(double t, const Eigen::Ref<const VectorXs>& p, const Eigen::Ref<const MatrixXs>& A,
                                Eigen::Ref<MatrixXs> out) const;

  virtual void multiplyDValueTransposeBy(double t, const Eigen::Ref<const VectorXs>& p,
                                         const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const;

 protected:
  using ControlParametrizationAbstractTpl<Scalar>::nu_;
  using ControlParametrizationAbstractTpl<Scalar>::np_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-two-rk3.hxx"

#endif  // CROCODDYL_CORE_CONTROLS_POLY_TWO_RK3_HPP_
