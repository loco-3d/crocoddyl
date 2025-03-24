///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_COSTS_RESIDUAL_COST_HPP_
#define CROCODDYL_CORE_COSTS_RESIDUAL_COST_HPP_

#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"

namespace crocoddyl {

/**
 * @brief Residual-based cost
 *
 * This cost function uses a residual model to compute the cost, i.e., \f[ cost
 * = a(\mathbf{r}(\mathbf{x}, \mathbf{u})), \f] where \f$\mathbf{r}(\cdot)\f$
 * and \f$a(\cdot)\f$ define the residual and activation functions,
 * respectively.
 *
 * Note that we only compute the Jacobians of the residual function. Therefore,
 * this cost model computes its Hessians through a Gauss-Newton approximation,
 * e.g., \f$\mathbf{l_{xu}} = \mathbf{R_x}^T \mathbf{A_{rr}} \mathbf{R_u} \f$,
 * where \f$\mathbf{R_x}\f$ and \f$\mathbf{R_u}\f$ are the Jacobians of the
 * residual function, and \f$\mathbf{A_{rr}}\f$ is the Hessian of the activation
 * model.
 *
 * As described in `CostModelAbstractTpl()`, the cost value and its derivatives
 * are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelResidualTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(CostModelBase, CostModelResidualTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataResidualTpl<Scalar> Data;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ResidualModelAbstractTpl<Scalar> ResidualModelAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the residual cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] residual    Residual model
   */
  CostModelResidualTpl(std::shared_ptr<typename Base::StateAbstract> state,
                       std::shared_ptr<ActivationModelAbstract> activation,
                       std::shared_ptr<ResidualModelAbstract> residual);
  /**
   * @brief Initialize the residual cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e.
   * \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state       State of the multibody system
   * @param[in] residual    Residual model
   */
  CostModelResidualTpl(std::shared_ptr<typename Base::StateAbstract> state,
                       std::shared_ptr<ResidualModelAbstract> residual);
  virtual ~CostModelResidualTpl() = default;

  /**
   * @brief Compute the residual cost
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<CostDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the residual cost based on state only
   *
   * It updates the total cost based on the state only. This function is used in
   * the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<CostDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the residual cost
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<CostDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the derivatives of the residual cost with respect to the
   * state only
   *
   * It updates the Jacobian and Hessian of the cost function based on the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<CostDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the residual cost data
   */
  virtual std::shared_ptr<CostDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the residual cost model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return CostModelResidualTpl<NewScalar> A cost model with the
   * new scalar type.
   */
  template <typename NewScalar>
  CostModelResidualTpl<NewScalar> cast() const;

  /**
   * @brief Print relevant information of the cost-residual model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;
};

template <typename _Scalar>
struct CostDataResidualTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  template <template <typename Scalar> class Model>
  CostDataResidualTpl(Model<Scalar>* const model,
                      DataCollectorAbstract* const data)
      : Base(model, data) {}
  virtual ~CostDataResidualTpl() = default;

  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::residual;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/costs/residual.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::CostModelResidualTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::CostDataResidualTpl)

#endif  // CROCODDYL_CORE_COSTS_RESIDUAL_COST_HPP_
