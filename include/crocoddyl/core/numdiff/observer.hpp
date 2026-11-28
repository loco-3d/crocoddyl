///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_OBSERVER_HPP_
#define CROCODDYL_CORE_NUMDIFF_OBSERVER_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/numdiff/restoration.hpp"
#include "crocoddyl/core/observer-base.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief Numerical differentiation wrapper for observer models.
 *
 * Computes Jacobians and Hessians of the dynamics, cost, and constraints via
 * finite differences, including parameter-related derivatives when `np > 0`.
 *
 * \sa `ObserverModelAbstractTpl`, `calcDiff()`
 */
template <typename _Scalar>
class ObserverModelNumDiffTpl : public ObserverModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, ObserverModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef ObserverModelAbstractTpl<Scalar> Base;
  typedef ObserverDataNumDiffTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the numdiff observer model
   *
   * @param[in] model              Observer model to differentiate
   * @param[in] with_gauss_approx  Use Gauss-Newton approximation for Hessians
   */
  explicit ObserverModelNumDiffTpl(std::shared_ptr<Base> model,
                                   bool with_gauss_approx = false);
  ObserverModelNumDiffTpl(std::shared_ptr<Base> model,
                          std::shared_ptr<ParameterManager> params,
                          bool with_gauss_approx = false);
  virtual ~ObserverModelNumDiffTpl() = default;

  /// @copydoc Base::calc(data, x, w)
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& w) override;

  /// @copydoc Base::calc(data, x)
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /// @copydoc Base::calcDiff(data, x, w)
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& w) override;

  /// @copydoc Base::calcDiff(data, x)
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /// @copydoc Base::createData()
  using Base::createData;
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /**
   * @brief Cast the numdiff observer model to a different scalar type.
   */
  template <typename NewScalar>
  ObserverModelNumDiffTpl<NewScalar> cast() const;

  /// @copydoc Base::quasiStatic()
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> w,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  /**
   * @brief Delegate update_tau to the wrapped model.
   */
  virtual void update_tau(const Eigen::Ref<const VectorXs>& tau_meas) override;

  /**
   * @brief Delegate update_p to the wrapped model.
   */
  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  virtual std::size_t get_ng() const override;
  virtual std::size_t get_nh() const override;
  virtual std::size_t get_ng_T() const override;
  virtual std::size_t get_nh_T() const override;
  virtual const VectorXs& get_g_lb() const override;
  virtual const VectorXs& get_g_ub() const override;

  /**
   * @brief Return the wrapped observer model.
   */
  const std::shared_ptr<Base>& get_model() const;

  const std::shared_ptr<ParameterManager>& get_params() const;

  /**
   * @brief Return the Jacobian finite-difference step.
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Set the Jacobian finite-difference step.
   */
  void set_disturbance(const Scalar disturbance);

  /**
   * @brief Whether the Gauss-Newton Hessian approximation is active.
   */
  bool get_with_gauss_approx();

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::np_;
  using Base::nu_;
  using Base::state_;
  using Base::tau_meas_;

 private:
  std::shared_ptr<Data> cast_data(
      const std::shared_ptr<ActionDataAbstract>& data) const;

  std::shared_ptr<Base> model_;
  std::shared_ptr<ParameterManager> params_;
  Scalar e_jac_;   //!< FD step for first-order derivatives
  Scalar e_hess_;  //!< FD step for second-order derivatives
  bool with_gauss_approx_;
};

template <typename _Scalar>
struct ObserverDataNumDiffTpl : public ObserverDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ObserverDataAbstractTpl<Scalar> Base;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ObserverDataNumDiffTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(model),
        params_data(params_data),
        dx(model->get_model()->get_state()->get_ndx()),
        dw(model->get_model()->get_nu()),
        dp(model->get_np()),
        p(model->get_np()),
        xp(model->get_model()->get_state()->get_nx()) {
    dx.setZero();
    dw.setZero();
    dp.setZero();
    p.setZero();
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
      data_w.push_back(create_wrapped_data());
    }
    for (std::size_t i = 0; i < np; ++i) {
      data_p.push_back(create_wrapped_data());
    }
  }
  virtual ~ObserverDataNumDiffTpl() = default;

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
      Ep.conservativeResize(1, model->get_np());
    }
    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t nu = model->get_model()->get_nu();
    const std::size_t np = model->get_np();

    dx.resize(ndx);
    dw.resize(nu);
    dp.resize(np);
    p.conservativeResize(np);
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
    while (data_w.size() < nu) {
      data_w.push_back(create_wrapped_data());
    }
    data_w.resize(nu);
    while (data_p.size() < np) {
      data_p.push_back(create_wrapped_data());
    }
    data_p.resize(np);

    dx.setZero();
    dw.setZero();
    dp.setZero();
    xp.setZero();
  }

  using Base::cost;
  using Base::dissipative_E;
  using Base::Ep;
  using Base::Eu;
  using Base::Ex;
  using Base::Fp;
  using Base::Fu;
  using Base::Fx;
  using Base::Gp;
  using Base::Hp;
  using Base::Lp;
  using Base::Lpp;
  using Base::Lpu;
  using Base::Lpx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::xnext;

  Scalar x_norm;   //!< Norm of the state vector
  Scalar xh_jac;   //!< FD step for \f$ \partial / \partial x \f$
  Scalar wh_jac;   //!< FD step for \f$ \partial / \partial w \f$
  Scalar ph_jac;   //!< FD step for \f$ \partial / \partial p \f$
  Scalar xh_hess;  //!< FD step for \f$ \partial^2 / \partial x^2 \f$
  Scalar wh_hess;  //!< FD step for \f$ \partial^2 / \partial w^2 \f$
  Scalar ph_hess;  //!< FD step for \f$ \partial^2 / \partial p^2 \f$
  Scalar xh_hess_pow2;
  Scalar wh_hess_pow2;
  Scalar ph_hess_pow2;
  Scalar xwh_hess_pow2;
  Scalar xph_hess_pow2;
  Scalar wph_hess_pow2;
  std::shared_ptr<ParameterDataManager> params_data;
  VectorXs dx;  //!< State disturbance
  VectorXs dw;  //!< Process-noise disturbance
  VectorXs dp;  //!< Parameter disturbance
  VectorXs p;   //!< Active parameter vector
  VectorXs xp;  //!< Integrated perturbed state

  std::shared_ptr<ActionDataAbstract> data_0;  //!< Baseline data
  std::vector<std::shared_ptr<ActionDataAbstract>>
      data_x;  //!< Data for state perturbations
  std::vector<std::shared_ptr<ActionDataAbstract>>
      data_w;  //!< Data for process-noise perturbations
  std::vector<std::shared_ptr<ActionDataAbstract>>
      data_p;  //!< Data for parameter perturbations
};

}  // namespace crocoddyl

#include "crocoddyl/core/numdiff/observer.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ObserverModelNumDiffTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ObserverDataNumDiffTpl)

#endif  // CROCODDYL_CORE_NUMDIFF_OBSERVER_HPP_
