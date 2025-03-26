///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, University of Trento,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
#define CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for an integrated action model
 *
 * An integrated action model is a special kind of action model that is obtained
 * by applying  a numerical integration scheme to a differential (i.e.,
 * continuous time) action model. Different integration schemes can be
 * implemented inheriting from this base class.
 *
 * The numerical integration introduces also the possibility to parametrize the
 * control trajectory inside an integration step, for instance using
 * polynomials. This requires introducing some notation to clarify the
 * difference between the control inputs of the differential model and the
 * control inputs to the integrated model. We have decided to use
 * \f$\mathbf{w}\f$ to refer to the control inputs of the differential model and
 * \f$\mathbf{u}\f$ for the control inputs of the integrated action model.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class IntegratedActionModelAbstractTpl
    : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataAbstractTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar>
      DifferentialActionModelAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar>
      ControlParametrizationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the integrator
   *
   * @param[in] model      Differential action model
   * @param[in] control    Control parametrization
   * @param[in] time_step  Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelAbstractTpl(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      std::shared_ptr<ControlParametrizationModelAbstract> control,
      const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);

  /**
   * @brief Initialize the integrator
   *
   * This initialization uses `ControlParametrizationPolyZeroTpl` for the
   * control parametrization.
   *
   * @param[in] model      Differential action model
   * @param[in] time_step  Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelAbstractTpl(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);
  virtual ~IntegratedActionModelAbstractTpl() = default;

  /**
   * @brief Create the integrator data
   *
   * @return the sympletic integrator data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;

  /**
   * @brief Return the number of inequality constraints
   */
  virtual std::size_t get_ng() const override;

  /**
   * @brief Return the number of equality constraints
   */
  virtual std::size_t get_nh() const override;

  /**
   * @brief Return the number of inequality terminal constraints
   */
  virtual std::size_t get_ng_T() const override;

  /**
   * @brief Return the number of equality terminal constraints
   */
  virtual std::size_t get_nh_T() const override;

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  virtual const VectorXs& get_g_lb() const override;

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  virtual const VectorXs& get_g_ub() const override;

  /**
   * @brief Return the differential action model associated to this integrated
   * action model
   */
  const std::shared_ptr<DifferentialActionModelAbstract>& get_differential()
      const;

  /**
   * @brief Return the control parametrization model associated to this
   * integrated action model
   */
  const std::shared_ptr<ControlParametrizationModelAbstract>& get_control()
      const;

  /**
   * @brief Return the time step used for the integration
   */
  const Scalar get_dt() const;

  /**
   * @brief Set the time step for the integration
   */
  void set_dt(const Scalar dt);

  DEPRECATED("The DifferentialActionModel should be set at construction time",
             void set_differential(
                 std::shared_ptr<DifferentialActionModelAbstract> model));

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control
                                    //!< limits are active
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Dimension of the control
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits

  IntegratedActionModelAbstractTpl()
      : differential_(nullptr),
        control_(nullptr),
        time_step_(0),
        time_step2_(0),
        with_cost_residual_(false) {}
  void init();

  std::shared_ptr<DifferentialActionModelAbstract>
      differential_;  //!< Differential action model that is integrated
  std::shared_ptr<ControlParametrizationModelAbstract>
      control_;  //!< Model of the control parametrization

  Scalar time_step_;   //!< Time step used for integration
  Scalar time_step2_;  //!< Square of the time step used for integration
  bool
      with_cost_residual_;  //!< Flag indicating whether a cost residual is used
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
  explicit IntegratedActionDataAbstractTpl(Model<Scalar>* const model)
      : Base(model) {}
  virtual ~IntegratedActionDataAbstractTpl() = default;

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

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::IntegratedActionModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::IntegratedActionDataAbstractTpl)

#endif  // CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
