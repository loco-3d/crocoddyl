///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_STATES_EUCLIDEAN_HPP_
#define CROCODDYL_CORE_STATES_EUCLIDEAN_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class StateVectorTpl : public StateAbstractTpl<_Scalar> {
public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit StateVectorTpl(const std::size_t &nx);
  virtual ~StateVectorTpl();

  virtual VectorXs zero() const;
  virtual VectorXs rand() const;
  virtual void diff(const Eigen::Ref<const VectorXs> &x0,
                    const Eigen::Ref<const VectorXs> &x1,
                    Eigen::Ref<VectorXs> dxout) const;
  virtual void integrate(const Eigen::Ref<const VectorXs> &x,
                         const Eigen::Ref<const VectorXs> &dx,
                         Eigen::Ref<VectorXs> xout) const;
  virtual void Jdiff(const Eigen::Ref<const VectorXs> &,
                     const Eigen::Ref<const VectorXs> &,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     const Jcomponent firstsecond = both) const;
  virtual void Jintegrate(const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &dx,
                          Eigen::Ref<MatrixXs> Jfirst,
                          Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both,
                          const AssignmentOp = setto) const;
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs> &x,
                                   const Eigen::Ref<const VectorXs> &dx,
                                   Eigen::Ref<MatrixXs> Jin,
                                   const Jcomponent firstsecond) const;

protected:
  using StateAbstractTpl<Scalar>::nx_;
  using StateAbstractTpl<Scalar>::ndx_;
  using StateAbstractTpl<Scalar>::nq_;
  using StateAbstractTpl<Scalar>::nv_;
  using StateAbstractTpl<Scalar>::lb_;
  using StateAbstractTpl<Scalar>::ub_;
  using StateAbstractTpl<Scalar>::has_limits_;
};

} // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/states/euclidean.hxx"

#endif // CROCODDYL_CORE_STATES_EUCLIDEAN_HPP_
