///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_STATE_HPP_
#define CROCODDYL_CORE_NUMDIFF_STATE_HPP_

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class StateNumDiffTpl : public StateAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<_Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit StateNumDiffTpl(boost::shared_ptr<Base> state);
  ~StateNumDiffTpl();

  virtual VectorXs zero() const;
  virtual VectorXs rand() const;
  virtual void diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                    Eigen::Ref<VectorXs> dxout) const;
  virtual void integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                         Eigen::Ref<VectorXs> xout) const;
  /**
   * @brief This computes the Jacobian of the diff method by finite
   * differentiation:
   * \f{equation}{
   *    Jfirst[:,k] = diff(int(x_1, dx_dist), x_2) - diff(x_1, x_2)/disturbance
   * \f}
   * and
   * \f{equation}{
   *    Jsecond[:,k] = diff(x_1, int(x_2, dx_dist)) - diff(x_1, x_2)/disturbance
   * \f}
   *
   * @param Jfirst
   * @param Jsecond
   * @param firstsecond
   */
  virtual void Jdiff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond, Jcomponent firstsecond = both) const;
  /**
   * @brief This computes the Jacobian of the integrate method by finite
   * differentiation:
   * \f{equation}{
   *    Jfirst[:,k] = diff( int(x, d_x), int( int(x, dx_dist), dx) )/disturbance
   * \f}
   * and
   * \f{equation}{
   *    Jsecond[:,k] = diff( int(x, d_x), int( x, dx + dx_dist) )/disturbance
   * \f}
   *
   * @param Jfirst
   * @param Jsecond
   * @param firstsecond
   */
  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                          Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                          Jcomponent firstsecond = both) const;
  const Scalar& get_disturbance() const;
  void set_disturbance(const Scalar& disturbance);

 private:
  /**
   * @brief This is the state we need to compute the numerical differentiation
   * from.
   */
  boost::shared_ptr<Base> state_;
  /**
   * @brief This the increment used in the finite differentiation and integration.
   */
  Scalar disturbance_;

 protected:
  using Base::has_limits_;
  using Base::lb_;
  using Base::ndx_;
  using Base::nq_;
  using Base::nv_;
  using Base::nx_;
  using Base::ub_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/state.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_STATE_HPP_
