///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/numdiff/restoration.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of an actuation
 * model.
 *
 * It computes the Jacobian of the residual model via numerical differentiation,
 * i.e., \f$\frac{\partial\boldsymbol{\tau}}{\partial\mathbf{x}}\f$ and
 * \f$\frac{\partial\boldsymbol{\tau}}{\partial\mathbf{u}}\f$ which denote the
 * Jacobians of the actuation function
 * \f$\boldsymbol{\tau}(\mathbf{x},\mathbf{u},\mathbf{p})\f$. With a
 * dynamics-parameter manager it also computes \f$d\tau/dp\f$ and friction
 * derivatives in the manager's active D075 layout. calc() establishes the
 * nominal point required by calcDiff(); terminal overloads retain the native
 * actuation lifecycle.
 *
 * \sa `ActuationModelAbstractTpl()`, `calcDiff()`
 */
template <typename _Scalar>
class ActuationModelNumDiffTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActuationModelBase, ActuationModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataNumDiffTpl<Scalar> Data;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the numdiff residual model
   *
   * @param model  Actuation model that we want to apply the numerical
   * differentiation
   */
  explicit ActuationModelNumDiffTpl(std::shared_ptr<Base> model);

  /**
   * @brief Initialize the numdiff actuation model with a parameter manager
   *
   * @param model   Actuation model that we want to apply the numerical
   * differentiation
   * @param params  Dynamics-parameter manager used for computing
   *                \f$\frac{\partial\boldsymbol{\tau}}{\partial\mathbf{p}}\f$
   */
  ActuationModelNumDiffTpl(std::shared_ptr<Base> model,
                           std::shared_ptr<ParameterManager> params);

  /**
   * @brief Destroy the numdiff actuation model
   */
  virtual ~ActuationModelNumDiffTpl() = default;

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const std::shared_ptr<ActuationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calc(const std::shared_ptr<ActuationDataAbstract>&
   * data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<ActuationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calcDiff(const
   * std::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc Base::commands()
   */
  virtual void commands(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& tau) override;

  /**
   * @brief @copydoc Base::torqueTransform()
   */
  virtual void torqueTransform(
      const std::shared_ptr<ActuationDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::createData()
   */
  virtual std::shared_ptr<ActuationDataAbstract> createData() override;

  /**
   * @brief Create the numdiff actuation data using an existing
   * parameter-manager data object
   */
  std::shared_ptr<ActuationDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& parameter_data);

  /** @brief Cast the wrapped model and manager to another scalar. */
  template <typename NewScalar>
  ActuationModelNumDiffTpl<NewScalar> cast() const;

  /**
   * @brief Return the original actuation model
   */
  const std::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the parameter manager used by the numerical
   * differentiation routine
   */
  const std::shared_ptr<ParameterManager>& get_params() const;

  /**
   * @brief Return the active parameter dimension used by the numdiff model
   */
  std::size_t get_np() const;

  /**
   * @brief Return the disturbance constant used by the numerical
   * differentiation routine
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Modify the disturbance constant used by the numerical
   * differentiation routine
   */
  void set_disturbance(const Scalar disturbance);

 private:
  std::shared_ptr<Base> model_;  //!< Actuation model being differentiated
  std::shared_ptr<ParameterManager> params_;  //!< Dynamics-parameter manager
  Scalar e_jac_;  //!< Constant used for computing disturbances in Jacobian
                  //!< calculation

 protected:
  using Base::nu_;
};

/**
 * @brief Data and preallocated scratch for ActuationModelNumDiffTpl.
 *
 * It owns independent nominal and perturbed actuation data. parameter_data
 * retains shared ownership of the manager payload used for parameter updates.
 */
template <typename _Scalar>
struct ActuationDataNumDiffTpl : public ActuationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ActuationDataAbstractTpl<Scalar> Base;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;

  /**
   * @brief Initialize the numdiff actuation data
   *
   * @tparam Model is the type of the `ActuationModelAbstractTpl`.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit ActuationDataNumDiffTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& parameter_data = nullptr)
      : Base(internal::checkNumDiffModel(model)),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        up(model->get_model()->get_nu()),
        dp(model->get_np()),
        p(model->get_np()),
        pp(model->get_np()),
        xp(model->get_model()->get_state()->get_nx()),
        friction_0(model->get_model()->get_state()->get_nv()),
        friction_p(model->get_model()->get_state()->get_nv()),
        dfriction_dx(model->get_model()->get_state()->get_nv(),
                     model->get_model()->get_state()->get_ndx()),
        dfriction_dp(model->get_model()->get_state()->get_nv(),
                     model->get_np()),
        dtau_dp(model->get_model()->get_state()->get_nv(), model->get_np()),
        parameter_data(parameter_data != nullptr
                           ? parameter_data
                           : model->get_params()->createData()) {
    dx.setZero();
    du.setZero();
    up.setZero();
    dp.setZero();
    p.setZero();
    pp.setZero();
    xp.setZero();
    friction_0.setZero();
    friction_p.setZero();
    dfriction_dx.setZero();
    dfriction_dp.setZero();
    dtau_dp.setZero();
    if (this->parameter_data == nullptr ||
        this->parameter_data->parameter_data != this->parameter_data.get() ||
        this->parameter_data->params == nullptr ||
        this->parameter_data->params->np != model->get_np()) {
      throw_pretty(
          "Invalid argument: parameter data has an incompatible "
          "dimension");
    }
    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t nu = model->get_model()->get_nu();
    const std::size_t np = model->get_np();
    data_0 = model->get_model()->createData();
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData());
    }
    for (std::size_t i = 0; i < np; ++i) {
      data_p.push_back(model->get_model()->createData());
    }
  }

  Scalar x_norm;  //!< Norm of the state vector
  Scalar
      xh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{x} \f$
  Scalar
      uh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{u} \f$
  Scalar
      ph_jac;  //!< Disturbance value used for computing \f$ \tau_\mathbf{p} \f$
  VectorXs dx;  //!< State disturbance
  VectorXs du;  //!< Control disturbance
  VectorXs up;  //!< Perturbed control vector
  VectorXs dp;  //!< Parameter disturbance
  VectorXs p;   //!< Nominal parameter vector
  VectorXs pp;  //!< Perturbed parameter vector
  VectorXs xp;  //!< The integrated state from the disturbance on one DoF "\f$
                //!< \int x dx_i \f$"
  VectorXs friction_0;    //!< Friction term at the nominal point
  VectorXs friction_p;    //!< Friction term at the disturbed point
  MatrixXs dfriction_dx;  //!< Partial derivatives of the friction term w.r.t.
                          //!< the state point
  MatrixXs dfriction_dp;  //!< Partial derivatives of the friction term w.r.t.
                          //!< the dynamics parameters
  MatrixXs dtau_dp;       //!< Partial derivatives of the actuation model w.r.t.
                          //!< the dynamics parameters
  std::shared_ptr<ParameterDataManager>
      parameter_data;            //!< Shared parameter-manager data
  std::shared_ptr<Base> data_0;  //!< The data that contains the final results
  std::vector<std::shared_ptr<Base> >
      data_x;  //!< The temporary data associated with the state variation
  std::vector<std::shared_ptr<Base> >
      data_u;  //!< The temporary data associated with the control variation
  std::vector<std::shared_ptr<Base> >
      data_p;  //!< The temporary data associated with the parameter variation

  using Base::dtau_du;
  using Base::dtau_dx;
  using Base::tau;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/actuation.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_
