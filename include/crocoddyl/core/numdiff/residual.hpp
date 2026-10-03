///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_RESIDUAL_HPP_
#define CROCODDYL_CORE_NUMDIFF_RESIDUAL_HPP_

#include <boost/function.hpp>

#include "crocoddyl/core/numdiff/restoration.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of a residual model.
 *
 * It computes the Jacobian of the residual model via numerical differentiation,
 * i.e., \f$\mathbf{R_x}\f$, \f$\mathbf{R_u}\f$ and \f$\mathbf{R_p}\f$ for
 * \f$\mathbf{r}(\mathbf{x},\mathbf{u},\mathbf{p})\f$. The parameter manager
 * supplies the active D075 layout, while registered reevaluation callbacks
 * refresh shared collectors before each state or control perturbation. All
 * nominal parameters and callback state are restored after success or failure.
 *
 * \sa `ResidualModelAbstractTpl()`, `calcDiff()`
 */
template <typename _Scalar>
class ResidualModelNumDiffTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataNumDiffTpl<Scalar> Data;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef boost::function<void(const VectorXs&, const VectorXs&)>
      ReevaluationFunction;

  /**
   * @brief Initialize the numdiff residual model
   *
   * @param model  Residual model that we want to apply the numerical
   * differentiation
   */
  explicit ResidualModelNumDiffTpl(const std::shared_ptr<Base>& model);
  ResidualModelNumDiffTpl(const std::shared_ptr<Base>& model,
                          std::shared_ptr<ParameterManager> params);

  /**
   * @brief Initialize the numdiff residual model
   */
  virtual ~ResidualModelNumDiffTpl() = default;

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calc(const std::shared_ptr<ResidualDataAbstract>&
   * data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calcDiff(const
   * std::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc Base::createData()
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;
  /** @brief Create data sharing an existing parameter-manager data object. */
  std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data,
      const std::shared_ptr<ParameterDataManager>& parameter_data);

  /** @brief Set the active parameter manager and initialize its data. */
  void set_params(const std::shared_ptr<ResidualDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params);

  /** @brief Update the nominal active parameter vector. */
  void update_p(const std::shared_ptr<ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& p);

  /** @brief Cast the wrapped residual and manager to another scalar. */
  template <typename NewScalar>
  ResidualModelNumDiffTpl<NewScalar> cast() const;

  /**
   * @brief Return the original residual model
   */
  const std::shared_ptr<Base>& get_model() const;

  /** @brief Return the active parameter manager. */
  const std::shared_ptr<ParameterManager>& get_params() const;

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

  /**
   * @brief Register functions that updates the shared data computed for a
   * system rollout The updated data is used to evaluate of the gradient and
   * Hessian.
   *
   * @param reevals are the registered functions.
   */
  void set_reevals(const std::vector<ReevaluationFunction>& reevals);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  /**
   * @brief Make sure that when we finite difference the residual model, the
   * user does not face unknown behaviour because of the finite differencing of
   * a quaternion around pi. This behaviour might occur if ResidualModelState
   * and FloatingInContact differential model are used together.
   *
   * For full discussions see issue
   * https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
   *
   * @param x is the state at which the check is performed.
   */
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/);
  void assertParameterData(
      const Data* const data,
      const std::shared_ptr<ParameterManager>& params) const;

  std::shared_ptr<Base> model_;  //!< Residual model being differentiated
  std::shared_ptr<ParameterManager> params_;
  Scalar e_jac_;  //!< Constant used for computing disturbances in Jacobian
                  //!< calculation
  std::vector<ReevaluationFunction>
      reevals_;  //!< Functions that needs execution before calc or calcDiff
};

/**
 * @brief Data and preallocated scratch for ResidualModelNumDiffTpl.
 *
 * The data owns independent residual data at the nominal and perturbed points.
 * An explicitly supplied parameter_data retains shared ownership. When it is
 * inferred from shared, it follows that collector's non-owning manager link;
 * shared and its manager data must then outlive this object.
 */
template <typename _Scalar>
struct ResidualDataNumDiffTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;

  /**
   * @brief Initialize the numdiff residual data
   *
   * @tparam Model is the type of the `ResidualModelAbstractTpl`.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit ResidualDataNumDiffTpl(
      Model<Scalar>* const model, DataCollectorAbstract* const shared_data,
      const std::shared_ptr<ParameterDataManager>& parameter_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(internal::checkNumDiffModel(model), shared_data),
        parameter_data(parameter_data),
        dx(model->get_state()->get_ndx()),
        xp(model->get_state()->get_nx()),
        du(model->get_nu()),
        up(model->get_nu()),
        dp(model->get_np()),
        p(model->get_np()),
        pp(model->get_np()) {
    dx.setZero();
    xp.setZero();
    du.setZero();
    up.setZero();
    dp.setZero();
    p.setZero();
    pp.setZero();
    DataCollectorParamsTpl<Scalar>* collector =
        dynamic_cast<DataCollectorParamsTpl<Scalar>*>(shared_data);
    if (collector != nullptr) {
      if (collector->params == nullptr) {
        throw_pretty("Invalid argument: collector parameter payload is null");
      }
      if (collector->parameter_data == nullptr) {
        throw_pretty("Invalid argument: collector parameter data is null");
      }
      if (collector->parameter_data->parameter_data !=
              collector->parameter_data ||
          collector->parameter_data->params != collector->params) {
        throw_pretty(
            "Invalid argument: collector parameter data is inconsistent");
      }
      if (this->parameter_data == nullptr) {
        this->parameter_data = std::shared_ptr<ParameterDataManager>(
            collector->parameter_data, [](ParameterDataManager*) {});
      } else if (this->parameter_data.get() != collector->parameter_data ||
                 this->parameter_data->params != collector->params) {
        throw_pretty(
            "Invalid argument: parameter data does not match the collector");
      }
    } else if (model->get_np() != 0) {
      throw_pretty("Invalid argument: shared data must provide parameter data");
    } else if (this->parameter_data == nullptr) {
      this->parameter_data = model->get_params()->createData();
    }

    if (this->parameter_data == nullptr ||
        this->parameter_data->parameter_data != this->parameter_data.get() ||
        this->parameter_data->params == nullptr ||
        this->parameter_data->params->np != model->get_np()) {
      throw_pretty(
          "Invalid argument: parameter data has an incompatible "
          "dimension");
    }

    const std::size_t& ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t& nu = model->get_model()->get_nu();
    const std::size_t& np = model->get_np();
    data_0 = model->get_model()->createData(shared_data);
    data_x.reserve(ndx);
    data_u.reserve(nu);
    data_p.reserve(np);
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData(shared_data));
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData(shared_data));
    }
    for (std::size_t i = 0; i < np; ++i) {
      data_p.push_back(model->get_model()->createData(shared_data));
    }
  }

  virtual ~ResidualDataNumDiffTpl() {}

  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  std::shared_ptr<ParameterDataManager>
      parameter_data;  //!< Parameter-manager data.
  Scalar x_norm;       //!< Norm of the state vector
  Scalar
      xh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{x} \f$
  Scalar
      uh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{u} \f$
  Scalar ph_jac;  //!< Disturbance value used for computing \f$ r_\mathbf{p} \f$
  VectorXs dx;    //!< State disturbance.
  VectorXs xp;    //!< The integrated state from the disturbance on one DoF "\f$
                  //!< \int x dx_i \f$".
  VectorXs du;    //!< Control disturbance.
  VectorXs up;  //!< The integrated control from the disturbance on one DoF "\f$
                //!< \int u du_i = u + du \f$".
  VectorXs dp;  //!< Parameter disturbance.
  VectorXs p;   //!< Nominal parameter vector.
  VectorXs pp;  //!< Perturbed parameter vector.
  std::shared_ptr<Base> data_0;  //!< The data at the approximation point.
  std::vector<std::shared_ptr<Base> >
      data_x;  //!< The temporary data associated with the state variation.
  std::vector<std::shared_ptr<Base> >
      data_u;  //!< The temporary data associated with the control variation.
  std::vector<std::shared_ptr<Base> >
      data_p;  //!< The temporary data associated with the parameter variation.
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/residual.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_RESIDUAL_HPP_
