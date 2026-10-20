///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEG_OBSERVER_BASE_HPP_
#define CROCODDYL_CORE_INTEG_OBSERVER_BASE_HPP_

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/data/observer.hpp"
#include "crocoddyl/core/observer-base.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for integrated observer models
 *
 * This class composes a continuous-time multibody dynamics model with observer
 * costs and optional observer constraints. It provides the shared-memory and
 * noise-projection utilities used by concrete observer integrators.
 */
template <typename _Scalar>
class IntegratedObserverModelAbstractTpl
    : public ObserverModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ObserverModelAbstractTpl<Scalar> Base;
  typedef IntegratedObserverDataAbstractTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ObserverDataAbstractTpl<Scalar> ObserverDataAbstract;
  typedef DynamicsModelAbstractTpl<Scalar> DynamicsModelAbstract;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DynamicsModelConstrainedForwardTpl<Scalar>
      DynamicsModelConstrainedForward;
  typedef DynamicsDataConstrainedForwardTpl<Scalar>
      DynamicsDataConstrainedForward;
  typedef DynamicsModelConstrainedInverseTpl<Scalar>
      DynamicsModelConstrainedInverse;
  typedef DynamicsDataConstrainedInverseTpl<Scalar>
      DynamicsDataConstrainedInverse;
  typedef ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  IntegratedObserverModelAbstractTpl(
      std::shared_ptr<DynamicsModelAbstract> dynamics,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr,
      const Scalar time_step = Scalar(1e-3));
  virtual ~IntegratedObserverModelAbstractTpl() = default;

  using Base::createData;

  void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  virtual void update_tau(const Eigen::Ref<const VectorXs>& tau_meas) override;

  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  virtual std::size_t get_ng() const override;

  virtual std::size_t get_nh() const override;

  virtual std::size_t get_ng_T() const override;

  virtual std::size_t get_nh_T() const override;

  virtual const VectorXs& get_g_lb() const override;

  virtual const VectorXs& get_g_ub() const override;

  const std::shared_ptr<DynamicsModelAbstract>& get_dynamics() const;

  const std::shared_ptr<CostModelSum>& get_costs() const;

  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;

  const std::shared_ptr<ParameterManager>& get_params() const;

  const std::shared_ptr<StateMultibody>& get_multibody_state() const;

  const std::shared_ptr<ImplicitConstraintModelMultiple>&
  get_dynamics_constraints() const;

  const Scalar get_dt() const;

  void set_dt(const Scalar dt);

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::ng_;
  using Base::ng_T_;
  using Base::nh_;
  using Base::nh_T_;
  using Base::np_;
  using Base::nr_;
  using Base::nu_;
  using Base::state_;
  using Base::tau_meas_;
  using Base::u_lb_;
  using Base::u_ub_;

  std::shared_ptr<Data> cast_data(
      const std::shared_ptr<ActionDataAbstract>& data) const;

  void refresh_constraint_bounds() const;

  std::shared_ptr<DynamicsDataConstrainedForward> get_forward_data(
      const std::shared_ptr<Data>& data) const;

  std::shared_ptr<DynamicsDataConstrainedInverse> get_inverse_data(
      const std::shared_ptr<Data>& data) const;

  std::size_t get_active_constraint_dim() const;

  MatrixXs& compute_noise_projector(const std::shared_ptr<Data>& data,
                                    const Eigen::Ref<const VectorXs>& x) const;

  VectorXs& compute_projected_noise(const std::shared_ptr<Data>& data,
                                    const Eigen::Ref<const VectorXs>& x,
                                    const Eigen::Ref<const VectorXs>& w) const;

  MatrixXs& compute_projected_noise_block_jacobian(
      const std::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& w_block,
      const Eigen::Ref<const VectorXs>& projected_block) const;

  MatrixXs& compute_projected_noise_jacobian(
      const std::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& w) const;

  std::shared_ptr<DynamicsModelAbstract> dynamics_;
  std::shared_ptr<CostModelSum> costs_;
  std::shared_ptr<ConstraintModelManager> constraints_;
  std::shared_ptr<ParameterManager> params_;
  std::shared_ptr<StateMultibody> state_multibody_;
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics_forward_;
  std::shared_ptr<DynamicsModelConstrainedInverse> dynamics_inverse_;
  std::shared_ptr<ImplicitConstraintModelMultiple> dynamics_constraints_;
  Scalar time_step_;
  Scalar time_step2_;
  VectorXs u_zero_;
  mutable VectorXs g_lb_live_;
  mutable VectorXs g_ub_live_;
};

template <typename _Scalar>
struct IntegratedObserverDataAbstractTpl
    : public ObserverDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ObserverDataAbstractTpl<Scalar> Base;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef DataCollectorObserverTpl<Scalar> DataCollectorObserver;
  typedef ImplicitConstraintDataMultipleTpl<Scalar>
      ImplicitConstraintDataMultiple;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedObserverDataAbstractTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>());
  virtual ~IntegratedObserverDataAbstractTpl() = default;

  template <class Model>
  void resize(Model* const model, const bool running_node = true) {
    Base::resize(model, running_node);
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nc_total =
        model->get_dynamics_constraints() != nullptr
            ? model->get_dynamics_constraints()->get_nc_total()
            : 0;

    dx.resize(ndx);
    projected_noise.resize(ndx);
    noise_projector.resize(ndx, ndx);
    dprojected_noise_dx.resize(ndx, ndx);
    Nc.resize(nv, nv);
    projected_noise_block_dx.resize(nv, nv);
    M.resize(nv, nv);
    Kinv.resize(nv + nc_total, nv + nc_total);
    dgrav_dq.resize(nv, nv);
    projector_x.resize(model->get_state()->get_nx());
    projector_zero.resize(nv);
    projector_rhs.resize(nv + nc_total);
    projector_sol.resize(nv + nc_total);
    projector_force.resize(nc_total);
    setZero();
  }

  virtual void setZero() override {
    Base::setZero();
    dx.setZero();
    projected_noise.setZero();
    noise_projector.setZero();
    dprojected_noise_dx.setZero();
    Nc.setZero();
    projected_noise_block_dx.setZero();
    M.setZero();
    Kinv.setZero();
    dgrav_dq.setZero();
    projector_x.setZero();
    projector_zero.setZero();
    projector_rhs.setZero();
    projector_sol.setZero();
    projector_force.setZero();
  }

  std::shared_ptr<DynamicsDataAbstract> dynamics;
  std::shared_ptr<CostDataSum> costs;
  std::shared_ptr<ConstraintDataManager> constraints;
  std::shared_ptr<ConstraintDataManager> constraints_terminal;
  VectorXs dx;
  VectorXs projected_noise;
  MatrixXs noise_projector;
  MatrixXs dprojected_noise_dx;
  MatrixXs Nc;
  MatrixXs projected_noise_block_dx;
  MatrixXs M;
  MatrixXs Kinv;
  MatrixXs dgrav_dq;
  pinocchio::DataTpl<Scalar> pinocchioW;
  std::shared_ptr<ImplicitConstraintDataMultiple> constraintsW;
  VectorXs projector_x;
  VectorXs projector_zero;
  VectorXs projector_rhs;
  VectorXs projector_sol;
  VectorXs projector_force;
};

}  // namespace crocoddyl

#include "crocoddyl/core/integ-observer-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::IntegratedObserverModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::IntegratedObserverDataAbstractTpl)

#endif  // CROCODDYL_CORE_INTEG_OBSERVER_BASE_HPP_
