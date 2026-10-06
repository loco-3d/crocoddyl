///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_DYNAMICS_IMPULSE_FORWARD_HPP_
#define CROCODDYL_MULTIBODY_DYNAMICS_IMPULSE_FORWARD_HPP_

#include <limits>

#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/data/implicit-constraints.hpp"
#include "crocoddyl/multibody/dynamics/dissipative.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Dynamics model for impulse forward dynamics in multibody systems.
 *
 * This model computes the post-impulse state \f$\mathbf{x}^+ =
 * [\mathbf{q}, \mathbf{v}^+]\f$ using the KKT system
 * \f[
 *   \left[\begin{matrix}\mathbf{M} & \mathbf{J}_c^\top \\ \mathbf{J}_c &
 *   \mathbf{0}\end{matrix}\right]
 *   \left[\begin{matrix}\mathbf{v}^+ \\ -\boldsymbol{\Lambda}
 *   \end{matrix}\right] =
 *   \left[\begin{matrix}\mathbf{M}\mathbf{v}^- \\
 *   -e\,\mathbf{J}_c\mathbf{v}^-\end{matrix}\right],
 * \f]
 * where \f$e\f$ is the restitution coefficient, \f$\mathbf{J}_c\f$ is
 * assembled from an `ImplicitConstraintModelMultipleTpl` stack, and
 * \f$\boldsymbol{\Lambda}\f$ are the impulse forces.
 *
 * Since the dynamics are intrinsically discrete the model is typed as
 * `DynamicsType::DiscreteTime` with \f$nu = 0\f$ (no control input).
 *
 * \sa `DynamicsModelAbstractTpl`, `calc()`, `calcDiff_xu()`, `createData()`
 */
template <typename _Scalar>
class DynamicsModelImpulseForwardTpl
    : public DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DynamicsModelBase, DynamicsModelImpulseForwardTpl)

  typedef _Scalar Scalar;
  typedef DynamicsModelAbstractTpl<Scalar> Base;
  typedef DynamicsDataImpulseForwardTpl<Scalar> Data;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the impulse forward dynamics model.
   *
   * @param[in] state        Multibody state
   * @param[in] constraints  Stack of implicit constraints (impulse contacts)
   * @param[in] np           Number of dynamics parameters
   * @param[in] r_coeff      Restitution coefficient (default 0)
   * @param[in] JMinvJt_damping  Damping of the operational-space inertia
   *                             (default 0)
   */
  DynamicsModelImpulseForwardTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ImplicitConstraintModelMultiple> constraints,
      const std::size_t np = 0, const Scalar r_coeff = Scalar(0.),
      const Scalar JMinvJt_damping = Scalar(0.));
  virtual ~DynamicsModelImpulseForwardTpl() = default;

  using Base::calc;
  using Base::calcDiff;
  using Base::calcDiff_xu;
  using Base::createData;

  virtual void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the post-impulse next state.
   *
   * Sets `data->vdot = [q; v+]` where `v+` is obtained from
   * `pinocchio::impulseDynamics`.
   *
   * @param[in] data  Impulse forward dynamics data
   * @param[in] x     State \f$\mathbf{x} = [\mathbf{q}, \mathbf{v}]\f$
   * @param[in] u     Unused (nu = 0)
   */
  virtual void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the Jacobians of the post-impulse next state.
   *
   * @param[in] data  Impulse forward dynamics data
   * @param[in] x     State \f$\mathbf{x} = [\mathbf{q}, \mathbf{v}]\f$
   * @param[in] u     Unused (nu = 0)
   */
  virtual void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                           const Eigen::Ref<const VectorXs>& x) override;

  virtual void calcDiff(const std::shared_ptr<DynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  virtual void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                           const Eigen::Ref<const VectorXs>& x,
                           const Eigen::Ref<const VectorXs>& u) override;

  virtual void calcDiff_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                          const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Allocate independent specialized data */
  virtual std::shared_ptr<DynamicsDataAbstract> createData() override;

  /** @brief Allocate data sharing a parameter-manager payload */
  virtual std::shared_ptr<DynamicsDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /** @brief Cast the complete model, constraints and parameters */
  template <typename NewScalar>
  DynamicsModelImpulseForwardTpl<NewScalar> cast() const;

  /** @brief Return whether data has the specialized runtime type */
  virtual bool checkData(
      const std::shared_ptr<DynamicsDataAbstract>& data) override;

  /** @brief Attach parameters and resize the supplied data */
  void set_params(const std::shared_ptr<DynamicsDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /** @brief Update active parameter values */
  void update_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& p) override;

  /** @brief Return the shared generic impulse-constraint stack */
  const std::shared_ptr<ImplicitConstraintModelMultiple>& get_constraints()
      const;

  /** @brief Return the shared parameter manager, or null when unset */
  const std::shared_ptr<ParameterManager>& get_params() const;

  /** @brief Return the Pinocchio model owned by the state */
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /** @brief Return the restitution coefficient */
  Scalar get_r_coeff() const;

  /** @brief Return the operational-space inertia damping */
  Scalar get_damping_factor() const;

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::np_;
  using Base::nu_;
  using Base::p_lb_;
  using Base::p_ub_;
  using Base::state_;
  using Base::tau_meas_;

 private:
  template <typename OtherScalar>
  friend class DynamicsModelImpulseForwardTpl;

  std::shared_ptr<ImplicitConstraintModelMultiple> constraints_;
  std::shared_ptr<ParameterManager> params_;
  pinocchio::ModelTpl<Scalar>* pinocchio_;
  Scalar r_coeff_;
  Scalar JMinvJt_damping_;
};

/**
 * @brief Data for impulse forward dynamics
 *
 * Owns Pinocchio, KKT and derivative workspaces. The embedded collector shares
 * joint, generic implicit-constraint and parameter payloads with downstream
 * consumers. External parameter data is retained through shared ownership;
 * `Base::shared` is a non-owning pointer to the embedded collector.
 */
template <typename _Scalar>
struct DynamicsDataImpulseForwardTpl : public DynamicsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsDataAbstractTpl<Scalar> Base;
  typedef JointDataAbstractTpl<Scalar> JointDataAbstract;
  typedef DataCollectorJointMultibodyInImplicitConstraintParamsTpl<Scalar>
      DataCollectorJointMultibodyInImplicitConstraintParams;
  typedef ActuationMultibodyParamsTpl<Scalar> ActuationMultibodyParams;
  typedef ParamsDataAbstractTpl<Scalar> ParamsDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef internal::DynamicsDataParameterRegressorTpl<Scalar>
      ParameterRegressorData;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

 private:
  template <template <typename Scalar> class Model>
  static Model<Scalar>* checkModel(Model<Scalar>* const model) {
    if (model == nullptr) {
      throw_pretty("Invalid argument: impulse forward model is null");
    }
    return model;
  }

  template <template <typename Scalar> class Model>
  void resizeParameterRegressor(Model<Scalar>* const model) {
    std::size_t nu = 0;
    const std::shared_ptr<ParameterManagerTpl<Scalar> > manager =
        model->get_params();
    if (manager != nullptr) {
      const typename ParameterManagerTpl<Scalar>::ParameterContainer&
          dynamics_params = manager->get_dynamics_params();
      for (typename ParameterManagerTpl<
               Scalar>::ParameterContainer::const_iterator it =
               dynamics_params.begin();
           it != dynamics_params.end(); ++it) {
        const std::shared_ptr<ParameterItemTpl<Scalar> >& item = it->second;
        if (!item->get_active()) {
          continue;
        }
        const std::shared_ptr<ActuationMultibodyParams> actuation_params =
            std::dynamic_pointer_cast<ActuationMultibodyParams>(
                item->get_param());
        if (actuation_params == nullptr) {
          continue;
        }
        const std::size_t item_nu = actuation_params->get_actuation()->get_nu();
        if (nu == 0) {
          nu = item_nu;
        } else if (nu != item_nu) {
          throw_pretty(
              "Invalid argument: impulse forward dynamics received "
              "actuation-parameter models with inconsistent nu");
        }
      }
    }
    parameter_regressor->resize(model, nu, &multibody);
  }

 public:
  /** @brief Allocate data, optionally sharing parameter-manager data */
  template <template <typename Scalar> class Model>
  explicit DynamicsDataImpulseForwardTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(checkModel(model)),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        params(params_data != nullptr
                   ? params_data
                   : (model->get_params() == nullptr
                          ? nullptr
                          : model->get_params()->createData())),
        external_params(params_data),
        shared_params(
            params_data != nullptr
                ? params_data->params
                : (params != nullptr
                       ? params->params
                       : std::allocate_shared<ParamsDataAbstract>(
                             Eigen::aligned_allocator<ParamsDataAbstract>(), 0,
                             model->get_np()))),
        joint(std::make_shared<JointDataAbstract>(model->get_state(), 0, 0)),
        multibody(&pinocchio, joint,
                  model->get_constraints()->createData(&pinocchio),
                  shared_params, params.get()),
        Kinv(model->get_state()->get_nv() +
                 model->get_constraints()->get_nc_total(),
             model->get_state()->get_nv() +
                 model->get_constraints()->get_nc_total()),
        df_dx(model->get_constraints()->get_nc_total(),
              model->get_state()->get_ndx()),
        df_du(model->get_constraints()->get_nc_total(), 0),
        dgrav_dq(model->get_state()->get_nv(), model->get_state()->get_nv()),
        vnone(model->get_state()->get_nv()),
        tmp_xparams(model->get_state()->get_nx()),
        tmp_dtau_dp(model->get_state()->get_nv(), model->get_np()),
        parameter_regressor(std::allocate_shared<ParameterRegressorData>(
            Eigen::aligned_allocator<ParameterRegressorData>(), model, 0,
            &multibody)) {
    Base::shared = &multibody;
    Kinv.setZero();
    df_dx.setZero();
    df_du.setZero();
    dgrav_dq.setZero();
    vnone.setZero();
    tmp_xparams.setZero();
    resizeParameterRegressor(model);
    tmp_dtau_dp.setZero();
    // q+ = q during an impulse, so the position block of Fx is identity
    Base::Fx
        .topLeftCorner(model->get_state()->get_nv(),
                       model->get_state()->get_nv())
        .setIdentity();
    if (params != nullptr) {
      shared_params = params->params;
      multibody.params = shared_params;
      multibody.parameter_data = params.get();
    }
  }

  /**
   * @brief Copy numerical storage and rebind the embedded collector
   *
   * Nested constraint, joint and parameter payloads preserve shared identity;
   * Pinocchio and all local derivative workspaces are independent.
   */
  DynamicsDataImpulseForwardTpl(const DynamicsDataImpulseForwardTpl& other)
      : Base(other),
        pinocchio(other.pinocchio),
        params(other.params),
        external_params(other.external_params),
        shared_params(other.shared_params),
        joint(other.joint),
        multibody(&pinocchio, joint, other.multibody.constraints, shared_params,
                  params.get()),
        Kinv(other.Kinv),
        df_dx(other.df_dx),
        df_du(other.df_du),
        dgrav_dq(other.dgrav_dq),
        vnone(other.vnone),
        tmp_xparams(other.tmp_xparams),
        tmp_dtau_dp(other.tmp_dtau_dp),
        parameter_regressor(std::allocate_shared<ParameterRegressorData>(
            Eigen::aligned_allocator<ParameterRegressorData>(),
            *other.parameter_regressor, &multibody)) {
    Base::shared = &multibody;
  }
  virtual ~DynamicsDataImpulseForwardTpl() = default;

  /** @brief Resize parameter-dependent storage after set_params() */
  template <template <typename Scalar> class Model>
  void resize(Model<Scalar>* const model) {
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nq = model->get_state()->get_nq();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t np = model->get_np();

    this->vdot.resize(nq + nv);
    this->Fx.resize(ndx, ndx);
    this->Fu.resize(ndx, 0);
    this->Fp.resize(ndx, np);
    this->dissipative_P.resize(1);
    this->dP_dv.resize(1, nv);
    this->dP_dp.resize(1, np);
    this->h.resize(0);
    this->Hx.resize(0, ndx);
    this->Hu.resize(0, 0);
    this->Hp.resize(0, np);
    this->g.resize(0);
    this->Gx.resize(0, ndx);
    this->Gu.resize(0, 0);
    this->Gp.resize(0, np);
    this->tmp_ustatic.resize(0);
    this->setZero();
    this->Fx.topLeftCorner(nv, nv).setIdentity();

    if (external_params != nullptr) {
      params = external_params;
      shared_params = params->params;
    } else if (model->get_params() == nullptr) {
      shared_params->resize(0, model->get_np());
      params = nullptr;
    } else {
      params = model->get_params()->createData();
      shared_params = params->params;
    }
    multibody.params = shared_params;
    multibody.parameter_data = params.get();
    resizeParameterRegressor(model);
    tmp_dtau_dp.resize(nv, np);
    tmp_dtau_dp.setZero();
  }

  pinocchio::DataTpl<Scalar> pinocchio;          //!< Owned Pinocchio data
  std::shared_ptr<ParameterDataManager> params;  //!< Active parameter data
  std::shared_ptr<ParameterDataManager>
      external_params;  //!< Externally supplied data, if any
  std::shared_ptr<ParamsDataAbstract> shared_params;  //!< Shared payload
  std::shared_ptr<JointDataAbstract> joint;           //!< Owned joint payload
  DataCollectorJointMultibodyInImplicitConstraintParams
      multibody;         //!< Collector exposed through Base::shared
  MatrixXs Kinv;         //!< Inverse impulse KKT matrix
  MatrixXs df_dx;        //!< Impulse-force state derivative
  MatrixXs df_du;        //!< nc_total x 0 — kept for updateForceDiff signature
  MatrixXs dgrav_dq;     //!< Gravity derivative workspace
  VectorXs vnone;        //!< Zero velocity passed to RNEA in calcDiff
  VectorXs tmp_xparams;  //!< Zero-velocity parameter state
  MatrixXs tmp_dtau_dp;  //!< Isolated inertial-torque parameter derivative
  std::shared_ptr<ParameterRegressorData>
      parameter_regressor;  //!< Internal continuous parameter view
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/dynamics/impulse-forward.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DynamicsModelImpulseForwardTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DynamicsDataImpulseForwardTpl)

#endif  // CROCODDYL_MULTIBODY_DYNAMICS_IMPULSE_FORWARD_HPP_
