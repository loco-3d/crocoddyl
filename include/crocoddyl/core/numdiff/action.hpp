///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTION_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/numdiff/restoration.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of an action model.
 *
 * It computes state, control and parameter derivatives of the dynamics, cost
 * and constraints. When requested, the cost Hessian uses the Gauss-Newton
 * approximation of the residual \f$\mathbf{r}(\mathbf{x},\mathbf{u},
 * \mathbf{p})\f$,
 * \f{eqnarray*}{
 *     \mathbf{\ell}_\mathbf{xx} &=& \mathbf{R_x}^T\mathbf{R_x} \\
 *     \mathbf{\ell}_\mathbf{uu} &=& \mathbf{R_u}^T\mathbf{R_u} \\
 *     \mathbf{\ell}_\mathbf{xu} &=& \mathbf{R_x}^T\mathbf{R_u}
 * \f}
 * for the state/control blocks. Parameter Hessians
 * \f$L_{pp},L_{px},L_{pu}\f$ are always evaluated directly by finite
 * differences. Without Gauss-Newton, every cost Hessian is evaluated directly.
 * An optional ParameterManager defines the active D075 layout; call update_p()
 * before calc()/calcDiff() when parameters are used. Running and terminal
 * calls keep the native ActionModelAbstract overload semantics.
 *
 * \sa `ActionModelAbstractTpl()`, `calcDiff()`
 */
template <typename _Scalar>
class ActionModelNumDiffTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, ActionModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataNumDiffTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  /**
   * @brief Initialize the numdiff action model
   *
   * @param[in] model              Action model that we want to apply the
   * numerical differentiation
   * @param[in] with_gauss_approx  True if we want to use the Gauss
   * approximation for computing the Hessians
   *
   * This legacy constructor differentiates only state and control. Use the
   * parameter-manager overload to differentiate active parameters.
   */
  explicit ActionModelNumDiffTpl(std::shared_ptr<Base> model,
                                 bool with_gauss_approx = false);

  /**
   * @brief Initialize with an active parameter manager
   *
   * @param[in] model Action model being differentiated
   * @param[in] params Manager defining the active parameter layout
   * @param[in] with_gauss_approx Use the residual Gauss-Newton Hessian
   */
  ActionModelNumDiffTpl(std::shared_ptr<Base> model,
                        std::shared_ptr<ParameterManager> params,
                        bool with_gauss_approx = false);
  virtual ~ActionModelNumDiffTpl() = default;

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calc(const std::shared_ptr<ActionDataAbstract>&
   * data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calcDiff(const std::shared_ptr<ActionDataAbstract>&
   * data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc Base::createData()
   */
  using Base::createData;
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /** @brief Set the active parameter manager on all internal data objects. */
  void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /** @brief Update the nominal active parameter vector. */
  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  /**
   * @brief @copydoc Base::quasiStatic()
   */
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  /**
   * @brief Cast the action numdiff model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ActionModelNumDiffTpl<NewScalar> An action model with the new
   * scalar type.
   */
  template <typename NewScalar>
  ActionModelNumDiffTpl<NewScalar> cast() const;

  /**
   * @brief Return the action model being differentiated
   */
  const std::shared_ptr<Base>& get_model() const;

  /** @brief Return the optional active parameter manager. */
  const std::shared_ptr<ParameterManager>& get_params() const;

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
   * @brief Print relevant information of the diff-action numdiff model
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

  std::shared_ptr<Base> model_;  //!< Action model being differentiated
  std::shared_ptr<ParameterManager> params_;  //!< Optional parameter manager
  Scalar e_jac_;   //!< Constant used for computing disturbances in Jacobian
                   //!< calculation
  Scalar e_hess_;  //!< Constant used for computing disturbances in Hessian
                   //!< calculation
  bool with_gauss_approx_;  //!< True if we want to use the Gauss approximation
                            //!< for computing the Hessians
};

/**
 * @brief Data and preallocated scratch for ActionModelNumDiffTpl.
 *
 * It owns independent wrapped data at the nominal and every perturbed point.
 * params_data retains shared ownership of an externally supplied parameter
 * payload; the remaining scratch is data-local and safe across model data
 * instances.
 */
template <typename _Scalar>
struct ActionDataNumDiffTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  /**
   * @brief Initialize the numdiff action data
   *
   * @tparam Model is the type of the `ActionModelAbstractTpl`.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit ActionDataNumDiffTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(internal::checkNumDiffModel(model)),
        params_data(params_data),
        Rx(model->get_model()->get_nr(),
           model->get_model()->get_state()->get_ndx()),
        Ru(model->get_model()->get_nr(), model->get_model()->get_nu()),
        dx(model->get_model()->get_state()->get_ndx()),
        dxn(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        up(model->get_model()->get_nu()),
        dp(model->get_np()),
        p(model->get_np()),
        pp(model->get_np()),
        xp(model->get_model()->get_state()->get_nx()) {
    Rx.setZero();
    Ru.setZero();
    dx.setZero();
    dxn.setZero();
    du.setZero();
    up.setZero();
    dp.setZero();
    p.setZero();
    pp.setZero();
    xp.setZero();
    if (params_data != nullptr &&
        (params_data->parameter_data != params_data.get() ||
         params_data->params == nullptr ||
         params_data->params->np != model->get_np())) {
      throw_pretty(
          "Invalid argument: parameter data has an incompatible "
          "dimension");
    }

    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t nu = model->get_model()->get_nu();
    const std::size_t np = model->get_np();
    const auto create_wrapped_data = [&]() {
      return params_data != nullptr
                 ? model->get_model()->createData(params_data)
                 : model->get_model()->createData();
    };
    data_0 = create_wrapped_data();
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(create_wrapped_data());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(create_wrapped_data());
    }
    for (std::size_t i = 0; i < np; ++i) {
      data_p.push_back(create_wrapped_data());
    }
  }

  template <class Model>
  void resize(Model* const model, const bool running_node = true) {
    if (running_node) {
      Base::resize(model, true);
    } else {
      const std::size_t ng_T = model->get_model()->get_ng_T();
      const std::size_t nh_T = model->get_model()->get_nh_T();
      this->g.conservativeResize(ng_T);
      this->Gx.conservativeResize(ng_T, model->get_state()->get_ndx());
      this->Gp.conservativeResize(ng_T, model->get_np());
      this->h.conservativeResize(nh_T);
      this->Hx.conservativeResize(nh_T, model->get_state()->get_ndx());
      this->Hp.conservativeResize(nh_T, model->get_np());
    }
    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t nu = model->get_model()->get_nu();
    const std::size_t np = model->get_np();

    Rx.resize(model->get_model()->get_nr(), ndx);
    Ru.resize(model->get_model()->get_nr(), nu);
    dx.resize(ndx);
    dxn.resize(ndx);
    du.resize(nu);
    up.resize(nu);
    dp.resize(np);
    p.conservativeResize(np);
    pp.resize(np);
    xp.resize(model->get_model()->get_state()->get_nx());

    const auto create_wrapped_data = [&]() {
      return params_data != nullptr
                 ? model->get_model()->createData(params_data)
                 : model->get_model()->createData();
    };
    if (data_0 == nullptr) {
      data_0 = create_wrapped_data();
    }
    while (data_x.size() < ndx) {
      data_x.push_back(create_wrapped_data());
    }
    data_x.resize(ndx);
    while (data_u.size() < nu) {
      data_u.push_back(create_wrapped_data());
    }
    data_u.resize(nu);
    while (data_p.size() < np) {
      data_p.push_back(create_wrapped_data());
    }
    data_p.resize(np);

    Rx.setZero();
    Ru.setZero();
    dx.setZero();
    dxn.setZero();
    du.setZero();
    up.setZero();
    dp.setZero();
    pp.setZero();
    xp.setZero();
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

  Scalar x_norm;  //!< Norm of the state vector
  Scalar
      xh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{x} \f$
  Scalar
      uh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{u} \f$
  Scalar
      ph_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{p} \f$
  Scalar xh_hess;  //!< Disturbance value used for computing \f$
                   //!< \ell_\mathbf{xx} \f$
  Scalar uh_hess;  //!< Disturbance value used for computing \f$
                   //!< \ell_\mathbf{uu} \f$
  Scalar ph_hess;  //!< Disturbance value used for computing \f$
                   //!< \ell_\mathbf{pp} \f$
  Scalar xh_hess_pow2;
  Scalar uh_hess_pow2;
  Scalar ph_hess_pow2;
  Scalar xuh_hess_pow2;
  Scalar xph_hess_pow2;
  Scalar uph_hess_pow2;
  std::shared_ptr<ParameterDataManager> params_data;  //!< Shared parameter data
  MatrixXs Rx;   //!< Cost residual jacobian: \f$ \frac{d r(x,u)}{dx} \f$
  MatrixXs Ru;   //!< Cost residual jacobian: \f$ \frac{d r(x,u)}{du} \f$
  VectorXs dx;   //!< State disturbance
  VectorXs dxn;  //!< Negative state disturbance scratch
  VectorXs du;   //!< Control disturbance
  VectorXs up;   //!< Perturbed control vector
  VectorXs dp;   //!< Parameter disturbance
  VectorXs p;    //!< Active parameter vector
  VectorXs pp;   //!< Perturbed parameter vector
  VectorXs xp;   //!< The integrated state from the disturbance on one DoF "\f$
                 //!< \int x dx_i \f$"
  std::shared_ptr<Base> data_0;  //!< The data that contains the final results
  std::vector<std::shared_ptr<Base> >
      data_x;  //!< The temporary data associated with the state variation
  std::vector<std::shared_ptr<Base> >
      data_u;  //!< The temporary data associated with the control variation
  std::vector<std::shared_ptr<Base> >
      data_p;  //!< The temporary data associated with the parameter variation
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/action.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActionModelNumDiffTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActionDataNumDiffTpl)

#endif  // CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
