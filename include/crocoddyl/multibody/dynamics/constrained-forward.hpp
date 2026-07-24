///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_DYNAMICS_CONSTRAINED_FORWARD_HPP_
#define CROCODDYL_MULTIBODY_DYNAMICS_CONSTRAINED_FORWARD_HPP_

#include <limits>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/data/implicit-constraints.hpp"
#include "crocoddyl/multibody/dynamics/dissipative.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Constrained multibody forward dynamics
 *
 * The model computes accelerations and constraint forces from an actuation
 * command (control mode) or measured torque (estimation mode). Active generic
 * implicit constraints define the KKT system; inactive constraints retain
 * allocated data but do not contribute. The running node exposes state,
 * control and parameter derivatives, while the terminal node only updates
 * multibody and constraint quantities.
 *
 * The state, actuation, constraint stack and optional parameter manager are
 * retained through shared ownership. calc() must precede calcDiff() for the
 * same data object.
 */
template <typename _Scalar>
class DynamicsModelConstrainedForwardTpl
    : public DynamicsModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DynamicsModelBase, DynamicsModelConstrainedForwardTpl)

  typedef _Scalar Scalar;
  typedef DynamicsModelAbstractTpl<Scalar> Base;
  typedef DynamicsDataConstrainedForwardTpl<Scalar> Data;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the constrained forward-dynamics model
   * @param[in] state Multibody state
   * @param[in] actuation Multibody actuation model
   * @param[in] implicit_constraints Generic implicit-constraint stack
   * @param[in] np Initial parameter dimension
   * @param[in] dyn_type Continuous control or estimation mode
   */
  DynamicsModelConstrainedForwardTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<ImplicitConstraintModelMultiple> implicit_constraints,
      const std::size_t np = 0,
      const DynamicsType dyn_type = DynamicsType::ContinuousControl);
  virtual ~DynamicsModelConstrainedForwardTpl() = default;

  using Base::calc;
  using Base::calcDiff;
  using Base::calcDiff_xu;
  using Base::createData;

  /** @brief Compute terminal multibody and constraint quantities */
  virtual void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /** @brief Compute running constrained acceleration, forces and power */
  virtual void calc(const std::shared_ptr<DynamicsDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Compute terminal state derivatives */
  virtual void calcDiff(const std::shared_ptr<DynamicsDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /** @brief Compute terminal state derivatives */
  virtual void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                           const Eigen::Ref<const VectorXs>& x) override;

  /** @brief Compute running state and control derivatives after calc() */
  virtual void calcDiff_xu(const std::shared_ptr<DynamicsDataAbstract>& data,
                           const Eigen::Ref<const VectorXs>& x,
                           const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Compute running parameter and dissipative-power derivatives */
  virtual void calcDiff_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                          const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Allocate an independent data object */
  virtual std::shared_ptr<DynamicsDataAbstract> createData() override;

  /** @brief Allocate data sharing an existing parameter-manager payload */
  virtual std::shared_ptr<DynamicsDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /** @brief Cast the complete model, constraints and parameters */
  template <typename NewScalar>
  DynamicsModelConstrainedForwardTpl<NewScalar> cast() const;

  /** @brief Return whether data has the specialized runtime type */
  virtual bool checkData(
      const std::shared_ptr<DynamicsDataAbstract>& data) override;

  /** @brief Compute a static actuation command under active constraints */
  virtual void quasiStatic(const std::shared_ptr<DynamicsDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  /** @brief Set measured torque used in estimation mode */
  virtual void update_tau(const Eigen::Ref<const VectorXs>& tau_meas) override;

  /** @brief Attach a parameter manager and resize the supplied data */
  void set_params(const std::shared_ptr<DynamicsDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /** @brief Update active parameter values in the supplied data */
  void update_p(const std::shared_ptr<DynamicsDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& p) override;

  /** @brief Return the shared actuation model */
  const std::shared_ptr<ActuationModelAbstract>& get_actuation() const;

  /** @brief Return the shared implicit-constraint stack */
  const std::shared_ptr<ImplicitConstraintModelMultiple>& get_constraints()
      const;

  /** @brief Return the shared parameter manager, or null when unset */
  const std::shared_ptr<ParameterManager>& get_params() const;

  /** @brief Return the Pinocchio model owned by the state */
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /** @brief Return the joint-armature vector */
  const VectorXs& get_armature() const;

  /** @brief Set the joint-armature vector */
  void set_armature(const VectorXs& armature);

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::dyn_type_;
  using Base::np_;
  using Base::nu_;
  using Base::p_lb_;
  using Base::p_ub_;
  using Base::state_;
  using Base::tau_meas_;

 private:
  template <typename OtherScalar>
  friend class DynamicsModelConstrainedForwardTpl;

  std::shared_ptr<ActuationModelAbstract> actuation_;
  std::shared_ptr<ImplicitConstraintModelMultiple> implicit_constraints_;
  std::shared_ptr<ParameterManager> params_;
  pinocchio::ModelTpl<Scalar>* pinocchio_;
  bool without_armature_;
  VectorXs armature_;
};

/**
 * @brief Data for constrained forward dynamics
 *
 * Owns Pinocchio and all numerical workspaces. The collector shares the
 * actuation, joint, implicit-constraint and parameter payloads used by
 * residuals and downstream models. An externally supplied parameter payload
 * is retained through shared ownership; `Base::shared` is a non-owning pointer
 * to the collector stored in this object.
 */
template <typename _Scalar>
struct DynamicsDataConstrainedForwardTpl
    : public DynamicsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsDataAbstractTpl<Scalar> Base;
  typedef JointDataAbstractTpl<Scalar> JointDataAbstract;
  typedef ParamsDataAbstractTpl<Scalar> ParamsDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef DataCollectorJointActMultibodyInImplicitConstraintParamsTpl<Scalar>
      DataCollectorJointActMultibodyInImplicitConstraintParams;
  typedef internal::DynamicsDataParameterRegressorTpl<Scalar>
      ParameterRegressorData;
  typedef ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

 private:
  template <template <typename Scalar> class Model>
  static Model<Scalar>* checkModel(Model<Scalar>* const model) {
    if (model == nullptr) {
      throw_pretty("Invalid argument: constrained forward model is null");
    }
    return model;
  }

 public:
  /** @brief Allocate data, optionally sharing a parameter-manager payload */
  template <template <typename Scalar> class Model>
  explicit DynamicsDataConstrainedForwardTpl(
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
                             Eigen::aligned_allocator<ParamsDataAbstract>(),
                             model->get_state(), 0, model->get_np()))),
        multibody(
            &pinocchio, model->get_actuation()->createData(),
            std::make_shared<JointDataAbstract>(
                model->get_state(), model->get_actuation(), model->get_nu()),
            model->get_constraints()->createData(&pinocchio), shared_params,
            params.get()),
        Kinv(model->get_state()->get_nv() +
                 model->get_constraints()->get_nc_total(),
             model->get_state()->get_nv() +
                 model->get_constraints()->get_nc_total()),
        df_dx(model->get_constraints()->get_nc_total(),
              model->get_state()->get_ndx()),
        df_du(model->get_constraints()->get_nc_total(),
              model->get_constraints()->get_nu()),
        tmp_xstatic(model->get_state()->get_nx()),
        tmp_Jstatic(model->get_state()->get_nv(),
                    model->get_nu() + model->get_constraints()->get_nc_total()),
        parameter_regressor(std::allocate_shared<ParameterRegressorData>(
            Eigen::aligned_allocator<ParameterRegressorData>(), model,
            model->get_actuation()->get_nu(), &multibody)) {
    Base::shared = &multibody;
    multibody.joint->dtau_du.diagonal().setOnes();
    pinocchio.lambda_c.resize(model->get_constraints()->get_nc_total());
    pinocchio.lambda_c.setZero();
    Kinv.setZero();
    df_dx.setZero();
    df_du.setZero();
    tmp_xstatic.setZero();
    tmp_Jstatic.setZero();
    if (params != nullptr) {
      shared_params = params->params;
      multibody.params = shared_params;
      multibody.parameter_data = params.get();
    }
  }

  /**
   * @brief Copy numerical storage and rebind the embedded collector
   *
   * Nested model data and parameter payloads preserve shared identity, while
   * Pinocchio and all local workspaces are copied independently.
   */
  DynamicsDataConstrainedForwardTpl(
      const DynamicsDataConstrainedForwardTpl& other)
      : Base(other),
        pinocchio(other.pinocchio),
        params(other.params),
        external_params(other.external_params),
        shared_params(other.shared_params),
        multibody(&pinocchio, other.multibody.actuation, other.multibody.joint,
                  other.multibody.constraints, shared_params, params.get()),
        Kinv(other.Kinv),
        df_dx(other.df_dx),
        df_du(other.df_du),
        tmp_xstatic(other.tmp_xstatic),
        tmp_Jstatic(other.tmp_Jstatic),
        parameter_regressor(std::allocate_shared<ParameterRegressorData>(
            Eigen::aligned_allocator<ParameterRegressorData>(),
            *other.parameter_regressor, &multibody)) {
    Base::shared = &multibody;
  }
  virtual ~DynamicsDataConstrainedForwardTpl() = default;

  /** @brief Resize parameter-dependent storage after set_params() */
  template <template <typename Scalar> class Model>
  void resize(Model<Scalar>* const model) {
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nq = model->get_state()->get_nq();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t np = model->get_np();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    const bool is_discrete =
        (model->get_dyn_type() == DynamicsType::DiscreteTime);
    const std::size_t nvdot = is_discrete ? nq + nv : nv;
    const std::size_t ndvdot = is_discrete ? ndx : nv;

    this->vdot.resize(nvdot);
    this->Fx.resize(ndvdot, ndx);
    this->Fu.resize(ndvdot, nu);
    this->Fp.resize(ndvdot, np);
    this->dissipative_P.resize(1);
    this->dP_dv.resize(1, nv);
    this->dP_dp.resize(1, np);
    this->h.resize(nh);
    this->Hx.resize(nh, ndx);
    this->Hu.resize(nh, nu);
    this->Hp.resize(nh, np);
    this->g.resize(ng);
    this->Gx.resize(ng, ndx);
    this->Gu.resize(ng, nu);
    this->Gp.resize(ng, np);
    this->tmp_ustatic.resize(nu);
    this->setZero();

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
    parameter_regressor->resize(model, model->get_actuation()->get_nu(),
                                &multibody);
  }

  pinocchio::DataTpl<Scalar> pinocchio;          //!< Owned Pinocchio data
  std::shared_ptr<ParameterDataManager> params;  //!< Active parameter data
  std::shared_ptr<ParameterDataManager>
      external_params;  //!< Externally supplied parameter data, if any
  std::shared_ptr<ParamsDataAbstract> shared_params;  //!< Shared payload
  DataCollectorJointActMultibodyInImplicitConstraintParams
      multibody;         //!< Collector exposed through Base::shared
  MatrixXs Kinv;         //!< Inverse KKT matrix
  MatrixXs df_dx;        //!< Constraint-force state derivative
  MatrixXs df_du;        //!< Constraint-force control derivative
  VectorXs tmp_xstatic;  //!< Quasi-static state workspace
  MatrixXs tmp_Jstatic;  //!< Quasi-static actuation/constraint map
  std::shared_ptr<ParameterRegressorData>
      parameter_regressor;  //!< Internal continuous parameter view
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/dynamics/constrained-forward.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DynamicsModelConstrainedForwardTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DynamicsDataConstrainedForwardTpl)

#endif  // CROCODDYL_MULTIBODY_DYNAMICS_CONSTRAINED_FORWARD_HPP_
