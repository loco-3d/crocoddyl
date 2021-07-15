///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree two, that is a quadratic function
 *
 * The size of the parameters p is 3 times the size of the control input u.
 * The first third of p represents the value of u at time 0.
 * The second third of p represents the value of u at time 0.5.
 * The last third of p represents the value of u at time 1.
 * This parametrization is suitable to be used with the RK-4 integration scheme,
 * because it requires the value of u exactly at 0, 0.5 and 1.
 */
template <typename _Scalar>
class ControlParametrizationModelPolyTwoRK4Tpl : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;

  explicit ControlParametrizationModelPolyTwoRK4Tpl(const std::size_t nu);
  virtual ~ControlParametrizationModelPolyTwoRK4Tpl();

  virtual void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                    const Eigen::Ref<const VectorXs>& p) const;

  virtual void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                      const Eigen::Ref<const VectorXs>& u) const;

  virtual void convert_bounds(const Eigen::Ref<const VectorXs>& u_lb, const Eigen::Ref<const VectorXs>& u_ub,
                              Eigen::Ref<VectorXs> p_lb, Eigen::Ref<VectorXs> p_ub) const;

  virtual void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                        const Eigen::Ref<const VectorXs>& p) const;

  virtual void multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>& p, const Eigen::Ref<const MatrixXs>& A,
                                  Eigen::Ref<MatrixXs> out) const;

  virtual void multiplyJacobianTransposeBy(double t, const Eigen::Ref<const VectorXs>& p,
                                           const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const;

 protected:
  using ControlParametrizationModelAbstractTpl<Scalar>::nu_;
  using ControlParametrizationModelAbstractTpl<Scalar>::np_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-two-rk4.hxx"

#endif  // CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_
