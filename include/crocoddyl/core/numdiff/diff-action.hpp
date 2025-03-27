///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          New York University, Max Planck Gesellschaft
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
#define CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_

#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of a differential
 * action model.
 *
 * It computes the Jacobian of the cost, its residual and dynamics via numerical
 * differentiation. It considers that the action model owns a cost residual and
 * the cost is the square of this residual, i.e.,
 * \f$\ell(\mathbf{x},\mathbf{u})=\frac{1}{2}\|\mathbf{r}(\mathbf{x},\mathbf{u})\|^2\f$,
 * where \f$\mathbf{r}(\mathbf{x},\mathbf{u})\f$ is the residual vector.  The
 * Hessian is computed only through the Gauss-Newton approximation, i.e.,
 * \f{eqnarray*}{
 *     \mathbf{\ell}_\mathbf{xx} &=& \mathbf{R_x}^T\mathbf{R_x} \\
 *     \mathbf{\ell}_\mathbf{uu} &=& \mathbf{R_u}^T\mathbf{R_u} \\
 *     \mathbf{\ell}_\mathbf{xu} &=& \mathbf{R_x}^T\mathbf{R_u}
 * \f}
 * where the Jacobians of the cost residuals are denoted by \f$\mathbf{R_x}\f$
 * and \f$\mathbf{R_u}\f$. Note that this approximation ignores the tensor
 * products (e.g., \f$\mathbf{R_{xx}}\mathbf{r}\f$).
 *
 * Finally, in the case that the cost does not have a residual, we set the
 * Hessian to zero, i.e., \f$\mathbf{L_{xx}} = \mathbf{L_{xu}} = \mathbf{L_{uu}}
 * = \mathbf{0}\f$.
 *
 * \sa `DifferentialActionModelAbstractTpl()`, `calcDiff()`
 */
template <typename _Scalar>
class DifferentialActionModelNumDiffTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DifferentialActionModelBase,
                         DifferentialActionModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataNumDiffTpl<Scalar> Data;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the numdiff differential action model
   *
   * @param[in] model              Differential action model that we want to
   * apply the numerical differentiation
   * @param[in] with_gauss_approx  True if we want to use the Gauss
   * approximation for computing the Hessians
   */
  explicit DifferentialActionModelNumDiffTpl(
      std::shared_ptr<Base> model, const bool with_gauss_approx = false);
  virtual ~DifferentialActionModelNumDiffTpl() = default;

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calc(const
   * std::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calcDiff(const
   * std::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc Base::createData()
   */
  virtual std::shared_ptr<DifferentialActionDataAbstract> createData() override;

  /**
   * @brief Cast the diff-action numdiff model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return DifferentialActionModelNumDiffTpl<NewScalar> A differential-action
   * model with the new scalar type.
   */
  template <typename NewScalar>
  DifferentialActionModelNumDiffTpl<NewScalar> cast() const;

  /**
   * @brief @copydoc Base::quasiStatic()
   */
  virtual void quasiStatic(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
      const std::size_t maxiter = 100,
      const Scalar tol = Scalar(1e-9)) override;

  /**
   * @brief Return the differential acton model that we use to numerical
   * differentiate
   */
  const std::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the disturbance constant used in the numerical
   * differentiation routine
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Modify the disturbance constant used in the numerical
   * differentiation routine
   */
  void set_disturbance(const Scalar disturbance);

  /**
   * @brief Identify if the Gauss approximation is going to be used or not.
   */
  bool get_with_gauss_approx();

  /**
   * @brief Print relevant information of the action numdiff model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control
                                    //!< limits
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits

 private:
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& x);
  std::shared_ptr<Base> model_;
  bool with_gauss_approx_;
  Scalar e_jac_;   //!< Constant used for computing disturbances in Jacobian
                   //!< calculation
  Scalar e_hess_;  //!< Constant used for computing disturbances in Hessian
                   //!< calculation
};

template <typename _Scalar>
struct DifferentialActionDataNumDiffTpl
    : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct a new ActionDataNumDiff object
   *
   * @tparam Model is the type of the ActionModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model),
        Rx(model->get_model()->get_nr(),
           model->get_model()->get_state()->get_ndx()),
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

  Scalar x_norm;  //!< Norm of the state vector
  Scalar
      xh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{x} \f$
  Scalar
      uh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{u} \f$
  Scalar xh_hess;  //!< Disturbance value used for computing \f$
                   //!< \ell_\mathbf{xx} \f$
  Scalar uh_hess;  //!< Disturbance value used for computing \f$
                   //!< \ell_\mathbf{uu} \f$
  Scalar xh_hess_pow2;
  Scalar uh_hess_pow2;
  Scalar xuh_hess_pow2;
  MatrixXs Rx;
  MatrixXs Ru;
  VectorXs dx;
  VectorXs du;
  VectorXs xp;
  std::shared_ptr<Base> data_0;
  std::vector<std::shared_ptr<Base> > data_x;
  std::vector<std::shared_ptr<Base> > data_u;

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::g;
  using Base::Gu;
  using Base::Gx;
  using Base::h;
  using Base::Hu;
  using Base::Hx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/diff-action.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DifferentialActionModelNumDiffTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DifferentialActionDataNumDiffTpl)

#endif  // CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
