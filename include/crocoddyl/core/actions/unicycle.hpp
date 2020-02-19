///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_
#define CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include <stdexcept>

namespace crocoddyl {
template <typename _Scalar>
class ActionModelUnicycleTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  
  ActionModelUnicycleTpl();
  ~ActionModelUnicycleTpl();

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const typename MathBase::VectorXs>& x,
            const Eigen::Ref<const typename MathBase::VectorXs>& u);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const typename MathBase::VectorXs>& x,
                const Eigen::Ref<const typename MathBase::VectorXs>& u);
  boost::shared_ptr<ActionDataAbstract> createData();

  const typename MathBase::Vector2s& get_cost_weights() const;
  void set_cost_weights(const typename MathBase::Vector2s& weights);
  
protected:
  using Base::nu_;                          //!< Control dimension
  using Base::nr_;                          //!< Dimension of the cost residual
  using Base::state_;  //!< Model of the state
  using Base::unone_;                   //!< Neutral state
  using Base::u_lb_;                    //!< Lower control limits
  using Base::u_ub_;                    //!< Upper control limits
  using Base::has_control_limits_;      //!< Indicates whether any of the control limits
  
 private:
  typename MathBase::Vector2s cost_weights_;
  Scalar dt_;
};

template <typename _Scalar>
struct ActionDataUnicycleTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  using Base::cost;
  using Base::xnext;
  using Base::r;
  using Base::Fx;
  using Base::Fu;
  using Base::Lx;
  using Base::Lu;
  using Base::Lxx;
  using Base::Lxu;
  using Base::Luu;
  
  template <typename Model>
  explicit ActionDataUnicycleTpl(Model* const model) : ActionDataAbstractTpl<Scalar>(model) {}

  
};

  
}  // namespace crocoddyl


/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/core/actions/unicycle.hxx>

#endif  // CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_
