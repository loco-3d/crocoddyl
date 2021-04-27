///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_COSTS_RESIDUAL_COST_HPP_
#define CROCODDYL_CORE_COSTS_RESIDUAL_COST_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/core/residual-base.hpp"

namespace crocoddyl {

/**
 * @brief Residual-based cost
 *
 * This cost function uses a residual model to compute the cost, i.e., \f[ cost = a(\mathbf{r}(\mathbf{x},
 * \mathbf{u})), \f] where \f$\mathbf{r}(\cdot)\f$ and \f$a(\cdot)\f$ define the residual and activation functions,
 * respectively.
 *
 * Note that we only compute the Jacobians of the residual function. Therefore, this cost model computes its Hessians
 * through a Gauss-Newton approximation, e.g., \f$\mathbf{l_{xu}} = \mathbf{R_x}^T \mathbf{A_{rr}} \mathbf{R_u} \f$,
 * where \f$\mathbf{R_x}\f$ and \f$\mathbf{R_u}\f$ are the Jacobians of the residual function, and
 * \f$\mathbf{A_{rr}}\f$ is the Hessian of the activation model.
 *
 * As described in `CostModelAbstractTpl()`, the cost value and its derivatives are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelResidualTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataResidualTpl<Scalar> Data;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ResidualModelAbstractTpl<Scalar> ResidualModelAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the residual cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] residual    Residual model
   * @param[in] activation  Activation model
   */
  CostModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                       boost::shared_ptr<ActivationModelAbstract> activation,
		       boost::shared_ptr<ResidualModelAbstract> residual);
  /**
   * @brief Initialize the residual cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state       State of the multibody system
   * @param[in] residual    Residual model
   */
  CostModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                       boost::shared_ptr<ResidualModelAbstract> residual);
  virtual ~CostModelResidualTpl();

  /**
   * @brief Compute the residual cost
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data,
		    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the residual cost
   *
   * @param[in] data  Residual cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
			const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the residual cost data
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

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
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataResidualTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data),
        Arr_Rx(model->get_residual()->get_nr(), model->get_state()->get_ndx()),
        Arr_Ru(model->get_residual()->get_nr(), model->get_nu()) {
    Arr_Rx.setZero();
    Arr_Ru.setZero();
  }

  MatrixXs Arr_Rx;
  MatrixXs Arr_Ru;
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

#endif  // CROCODDYL_CORE_COSTS_RESIDUAL_COST_HPP_
