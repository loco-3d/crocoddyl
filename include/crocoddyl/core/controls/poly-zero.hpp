///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree zero, that is a constant
 */
template <typename _Scalar>
class ControlParametrizationModelPolyZeroTpl : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;

  explicit ControlParametrizationModelPolyZeroTpl(const std::size_t nu);
  virtual ~ControlParametrizationModelPolyZeroTpl();

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
#include "crocoddyl/core/controls/poly-zero.hxx"

#endif  // CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_
