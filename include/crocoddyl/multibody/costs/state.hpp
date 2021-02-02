///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_STATE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief State cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{x}\ominus\mathbf{x}^*\f$, where
 * \f$\mathbf{x},\mathbf{x}^*\in~\mathcal{X}\f$ are the current and reference states, respectively, which belong to the
 * state manifold \f$\mathcal{X}\f$. Note that the dimension of the residual vector is obtained from
 * `StateAbstract::get_ndx()`.
 *
 * Both cost and residual derivatives are computed analytically.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xx}} = \mathbf{l_{x}}^T \mathbf{l_{x}} \f$.
 *
 * As described in CostModelAbstractTpl(), the cost value and its derivatives are calculated by `calc` and `calcDiff`,
 * respectively.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelStateTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelStateTpl<Scalar> ResidualModelState;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the state cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] xref        Reference state
   * @param[in] nu          Dimension of the control vector
   */
  CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                    boost::shared_ptr<ActivationModelAbstract> activation, const VectorXs& xref, const std::size_t nu);

  /**
   * @brief Initialize the state cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] xref        Reference state
   */
  CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                    boost::shared_ptr<ActivationModelAbstract> activation, const VectorXs& xref);

  /**
   * @brief Initialize the state cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] xref   Reference state
   * @param[in] nu     Dimension of the control vector
   */
  CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const VectorXs& xref, const std::size_t nu);

  /**
   * @brief Initialize the state cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] xref   Reference state
   */
  CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const VectorXs& xref);

  /**
   * @brief Initialize the state cost model
   *
   * The default reference state is obtained from `StateAbstractTpl::zero()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] nu          Dimension of the control vector
   */
  CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                    boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t nu);

  /**
   * @brief Initialize the state cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * The default reference state is obtained from `StateAbstractTpl::zero()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu     Dimension of the control vector
   */
  CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu);

  /**
   * @brief Initialize the state cost model
   *
   * The default state reference is obtained from `StateAbstractTpl::zero()`, and `nu` from
   * `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   */
  CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                    boost::shared_ptr<ActivationModelAbstract> activation);

  /**
   * @brief Initialize the state cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * The default state reference is obtained from `StateAbstractTpl::zero()` and `nu` from
   * `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   */
  explicit CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state);
  virtual ~CostModelStateTpl();

  /**
   * @brief Compute the state cost
   *
   * @param[in] data  State cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the state cost
   *
   * @param[in] data  State cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the state cost data
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Return the state reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  /**
   * @brief Modify the state reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  VectorXs xref_;                                                         //!< Reference state
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct CostDataStateTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataStateTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Rx(model->get_activation()->get_nr(), model->get_state()->get_ndx()) {
    Arr_Rx.setZero();
  }

  MatrixXs Arr_Rx;

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
#include "crocoddyl/multibody/costs/state.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
