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

/**
 * @brief Abstract class for an integrated action model
 *
 * An integrated action model is a special kind of action model that is obtained by applying
 * a numerical integration scheme to a differential (i.e. continuous time) action model.
 * Different integration schemes can be implemented inheriting from this base class.
 *
 * The numerical integration introduces also the possibility to parametrize the control
 * trajectory inside an integration step, for instance using polynomials. This requires
 * introducing some notation to clarify the difference between the control inputs of
 * the differential model and the control inputs to the integrated model. We have decided
 * to use u_diff and u_params to refer to the control inputs of the differential and integrated
 * action model, respectively. Since these names are much longer than the classic one-letter
 * names, they make the names of the derivative variables hard to read. Therefore, we have
 * decided to introduce also a 1-letter version for these names, that is u for u_params
 * and w for u_diff.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
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
  DEPRECATED("The DifferentialActionModel should be set at construction time",
             void set_differential(boost::shared_ptr<DifferentialActionModelAbstract> model));

  /**
   * @brief Return the dimension of the control input of the differential action model
   */
  std::size_t get_nu_diff() const;

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits are active
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Dimension of the control
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
#include "crocoddyl/core/integ-action-base.hxx"

#endif  // CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
