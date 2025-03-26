///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_STATES_EUCLIDEAN_HPP_
#define CROCODDYL_CORE_STATES_EUCLIDEAN_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class StateVectorTpl : public StateAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(StateBase, StateVectorTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit StateVectorTpl(const std::size_t nx);
  virtual ~StateVectorTpl();

  virtual VectorXs zero() const override;
  virtual VectorXs rand() const override;
  virtual void diff(const Eigen::Ref<const VectorXs>& x0,
                    const Eigen::Ref<const VectorXs>& x1,
                    Eigen::Ref<VectorXs> dxout) const override;
  virtual void integrate(const Eigen::Ref<const VectorXs>& x,
                         const Eigen::Ref<const VectorXs>& dx,
                         Eigen::Ref<VectorXs> xout) const override;
  virtual void Jdiff(const Eigen::Ref<const VectorXs>&,
                     const Eigen::Ref<const VectorXs>&,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     const Jcomponent firstsecond = both) const override;
  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& dx,
                          Eigen::Ref<MatrixXs> Jfirst,
                          Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both,
                          const AssignmentOp = setto) const override;
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs>& x,
                                   const Eigen::Ref<const VectorXs>& dx,
                                   Eigen::Ref<MatrixXs> Jin,
                                   const Jcomponent firstsecond) const override;

  template <typename NewScalar>
  StateVectorTpl<NewScalar> cast() const;

  /**
   * @brief Print relevant information of the state vector
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using StateAbstractTpl<Scalar>::nx_;
  using StateAbstractTpl<Scalar>::ndx_;
  using StateAbstractTpl<Scalar>::nq_;
  using StateAbstractTpl<Scalar>::nv_;
  using StateAbstractTpl<Scalar>::lb_;
  using StateAbstractTpl<Scalar>::ub_;
  using StateAbstractTpl<Scalar>::has_limits_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/states/euclidean.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::StateVectorTpl)

#endif  // CROCODDYL_CORE_STATES_EUCLIDEAN_HPP_
