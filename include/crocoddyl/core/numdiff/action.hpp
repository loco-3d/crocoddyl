///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTION_HPP_

#include <vector>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of an action model.
 *
 * It computes the Jacobian of the cost, its residual and dynamics via numerical differentiation.
 * It considers that the action model owns a cost residual and the cost is the square of this residual, i.e.,
 * \f$\ell(\mathbf{x},\mathbf{u})=\frac{1}{2}\|\mathbf{r}(\mathbf{x},\mathbf{u})\|^2\f$, where
 * \f$\mathbf{r}(\mathbf{x},\mathbf{u})\f$ is the residual vector.  The Hessian is computed only through the
 * Gauss-Newton approximation, i.e.,
 * \f{eqnarray*}{
 *     \mathbf{\ell}_\mathbf{xx} &=& \mathbf{R_x}^T\mathbf{R_x} \\
 *     \mathbf{\ell}_\mathbf{uu} &=& \mathbf{R_u}^T\mathbf{R_u} \\
 *     \mathbf{\ell}_\mathbf{xu} &=& \mathbf{R_x}^T\mathbf{R_u}
 * \f}
 * where the Jacobians of the cost residuals are denoted by \f$\mathbf{R_x}\f$ and \f$\mathbf{R_u}\f$.
 * Note that this approximation ignores the tensor products (e.g., \f$\mathbf{R_{xx}}\mathbf{r}\f$).
 *
 * Finally, in the case that the cost does not have a residual, we set the Hessian to zero, i.e.,
 * \f$\mathbf{L_{xx}} = \mathbf{L_{xu}} = \mathbf{L_{uu}} = \mathbf{0}\f$.
 *
 * \sa `ActionModelAbstractTpl()`, `calcDiff()`
 */
template <typename _Scalar>
class ActionModelNumDiffTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataNumDiffTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  /**
   * @brief Initialize the numdiff action model
   *
   * @param[in] model              Action model that we want to apply the numerical differentiation
   * @param[in] with_gauss_approx  True if we want to use the Gauss approximation for computing the Hessians
   */
  explicit ActionModelNumDiffTpl(boost::shared_ptr<Base> model, bool with_gauss_approx = false);
  virtual ~ActionModelNumDiffTpl();

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc Base::createData()
   */
  virtual boost::shared_ptr<ActionDataAbstract> createData();

  /**
   * @brief Return the acton model that we use to numerical differentiate
   */
  const boost::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the disturbance used in the numerical differentiation routine
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Modify the disturbance used in the numerical differentiation routine
   */
  void set_disturbance(const Scalar disturbance);

  /**
   * @brief Identify if the Gauss approximation is going to be used or not.
   */
  bool get_with_gauss_approx();

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state

 private:
  /**
   * @brief Make sure that when we finite difference the Action Model, the user
   * does not face unknown behaviour because of the finite differencing of a
   * quaternion around pi. This behaviour might occur if CostModelState and
   * FloatingInContact differential model are used together.
   *
   * For full discussions see issue
   * https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
   *
   * @param x is the state at which the check is performed.
   */
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& x);

  boost::shared_ptr<Base> model_;  //!< Action model hat we want to apply the numerical differentiation
  Scalar disturbance_;             //!< Disturbance used in the numerical differentiation routine
  bool with_gauss_approx_;         //!< True if we want to use the Gauss approximation for computing the Hessians
};

template <typename _Scalar>
struct ActionDataNumDiffTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  /**
   * @brief Initialize the numdiff action data
   *
   * @tparam Model is the type of the `ActionModelAbstractTpl`.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit ActionDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model),
        Rx(model->get_model()->get_nr(), model->get_model()->get_state()->get_ndx()),
        Ru(model->get_model()->get_nr(), model->get_model()->get_nu()),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        xp(model->get_model()->get_state()->get_nx()) {
    Rx.setZero();
    Ru.setZero();
    dx.setZero();
    du.setZero();
    xp.setZero();

    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData();
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData());
    }
  }

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

  MatrixXs Rx;                     //!< Cost residual jacobian: \f$ \frac{d r(x,u)}{dx} \f$
  MatrixXs Ru;                     //!< Cost residual jacobian: \f$ \frac{d r(x,u)}{du} \f$
  VectorXs dx;                     //!< State disturbance
  VectorXs du;                     //!< Control disturbance
  VectorXs xp;                     //!< The integrated state from the disturbance on one DoF "\f$ \int x dx_i \f$"
  boost::shared_ptr<Base> data_0;  //!< The data that contains the final results
  std::vector<boost::shared_ptr<Base> > data_x;  //!< The temporary data associated with the state variation
  std::vector<boost::shared_ptr<Base> > data_u;  //!< The temporary data associated with the control variation
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/action.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
