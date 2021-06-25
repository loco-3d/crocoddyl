///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Oxford,
//                     University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
#define CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class IntegratedActionModelAbstractTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataEulerTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar> DifferentialActionModelAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> ControlParametrizationModelAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  IntegratedActionModelAbstractTpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                                  const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);
  IntegratedActionModelAbstractTpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                                  boost::shared_ptr<ControlParametrizationModelAbstract> control,
                                  const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);
  virtual ~IntegratedActionModelAbstractTpl();

  const boost::shared_ptr<DifferentialActionModelAbstract>& get_differential() const;
  const Scalar get_dt() const;

  void set_dt(const Scalar dt);
  // void set_differential(boost::shared_ptr<DifferentialActionModelAbstract> model);

  /**
   * @brief Return the dimension of the control input of the differential action model
   */
  std::size_t get_nu_diff() const;

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits are active
  using Base::nu_;                  //!< Dimension of the control
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state

  void init();
  
  boost::shared_ptr<ControlParametrizationModelAbstract> control_;  //!< Model of the control discretization
  boost::shared_ptr<ControlParametrizationDataAbstract> controlData_;
  boost::shared_ptr<DifferentialActionModelAbstract> differential_;
  Scalar time_step_;
  Scalar time_step2_;
  bool with_cost_residual_;
  bool enable_integration_;
};

template <typename _Scalar>
struct IntegratedActionDataAbstractTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataAbstractTpl(Model<Scalar>* const model) : Base(model) {}
  virtual ~IntegratedActionDataAbstractTpl() {}

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/integr-action-base.hxx"

#endif  // CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
