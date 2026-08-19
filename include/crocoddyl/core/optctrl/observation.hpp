///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OPTCTRL_OBSERVATION_HPP_
#define CROCODDYL_CORE_OPTCTRL_OBSERVATION_HPP_

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/observer-base.hpp"
#include "crocoddyl/core/optctrl/problem-abstract.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief Optimal-estimation problem with measured torques and shared
 * parameters
 *
 * An observation problem estimates a state trajectory
 * \f$\{\mathbf{x}_t\}\f$ using process-noise controls
 * \f$\{\mathbf{w}_t\}\f$, measured torques and model parameters. The running
 * models can form either a single phase or multiple phases. Within each phase,
 * all observer data share one `ParameterManagerTpl` and one
 * `ParameterDataManagerTpl`. The terminal node uses the parameters of the last
 * phase. Models and parameter managers are shared, while trajectory data and
 * phase containers are owned by the problem.
 *
 * Parameter status/layout changes invalidate existing data and require
 * reconstruction of the problem. Measured torque belongs to each observer
 * model, so distinct node measurements require distinct model instances. The
 * frozen donor API intentionally exposes no structural mutation operations;
 * construct a new observation problem when its model layout changes.
 */
template <typename _Scalar>
class ObservationProblemTpl : public ProblemAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ProblemAbstractTpl<Scalar> Base;
  typedef ObserverModelAbstractTpl<Scalar> ObserverModelAbstract;
  typedef ActionModelAbstractTpl<Scalar> ActionModelAbstract;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Construct a multi-phase observation problem
   *
   * @param[in] x0 Initial state
   * @param[in] tau_meas Measured torques at all running nodes
   * @param[in] model_phases Running observer models grouped by phase
   * @param[in] terminal_model Terminal observer model
   * @param[in] params One parameter manager per phase
   */
  ObservationProblemTpl(
      const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
      const std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >&
          model_phases,
      std::shared_ptr<ObserverModelAbstract> terminal_model,
      const std::vector<std::shared_ptr<ParameterManager> >& params);

  /** @brief Construct a constrained multi-phase observation problem */
  ObservationProblemTpl(
      const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
      const std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >&
          model_phases,
      std::shared_ptr<ObserverModelAbstract> terminal_model,
      const std::vector<std::shared_ptr<ParameterManager> >& params,
      const std::vector<std::shared_ptr<ConstraintModelManager> >&
          parameter_constraints);

  /**
   * @brief Construct a single-phase observation problem
   *
   * @param[in] x0 Initial state
   * @param[in] tau_meas Measured torques at all running nodes
   * @param[in] running_models Running observer models
   * @param[in] terminal_model Terminal observer model
   * @param[in] params Parameter manager shared by the phase
   */
  ObservationProblemTpl(
      const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
      const std::vector<std::shared_ptr<ObserverModelAbstract> >&
          running_models,
      std::shared_ptr<ObserverModelAbstract> terminal_model,
      std::shared_ptr<ParameterManager> params);

  /** @brief Construct a constrained single-phase observation problem */
  ObservationProblemTpl(
      const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
      const std::vector<std::shared_ptr<ObserverModelAbstract> >&
          running_models,
      std::shared_ptr<ObserverModelAbstract> terminal_model,
      std::shared_ptr<ParameterManager> params,
      std::shared_ptr<ConstraintModelManager> parameter_constraints);

  virtual ~ObservationProblemTpl() = default;

  /** @brief Evaluate running and terminal observer models and total cost */
  virtual Scalar calc(const std::vector<VectorXs>& xs,
                      const std::vector<VectorXs>& us) override;

  /** @brief Evaluate derivatives and return the current total cost */
  virtual Scalar calcDiff(const std::vector<VectorXs>& xs,
                          const std::vector<VectorXs>& us) override;

  /** @brief Roll out process-noise controls from the stored initial state */
  virtual void rollout(const std::vector<VectorXs>& us,
                       std::vector<VectorXs>& xs) override;

  /** @brief Return the number of running nodes */
  virtual std::size_t get_T() const override;

  /** @brief Return the initial state */
  virtual const VectorXs& get_x0() const override;

  /** @brief Return the state dimension */
  virtual std::size_t get_nx() const override;

  /** @brief Return the state tangent dimension */
  virtual std::size_t get_ndx() const override;

  /** @brief Observation problems evaluate serially */
  virtual std::size_t get_nthreads() const override;

  /** @brief Return the flattened running observer models */
  virtual const std::vector<std::shared_ptr<ActionModelAbstract> >&
  get_runningModels() const override;

  /** @brief Return the terminal observer model */
  virtual const std::shared_ptr<ActionModelAbstract>& get_terminalModel()
      const override;

  /** @brief Return the running observer data */
  virtual const std::vector<std::shared_ptr<ActionDataAbstract> >&
  get_runningDatas() const override;

  /** @brief Return the terminal observer data */
  virtual const std::shared_ptr<ActionDataAbstract>& get_terminalData()
      const override;

  /** @brief Return and clear the structural-update flag */
  virtual bool is_updated() override;

  /** @brief Set the structural-update flag */
  virtual void set_is_updated(const bool val) override;

  /** @brief Return the number of parameterized phases */
  virtual std::size_t get_n_phases() const override;

  /** @brief Update one phase parameter payload exactly once */
  virtual void update_p(const Eigen::Ref<const VectorXs>& p,
                        const std::size_t phase_idx = 0) override;

  /** @brief Update the measured torque of one running observer model */
  void update_tau(const std::size_t t,
                  const Eigen::Ref<const VectorXs>& tau_meas);

  /** @brief Update measured torques at all running nodes */
  void update_us(const std::vector<VectorXs>& tau_meas);

  /** @brief Return the observer models in one phase */
  std::vector<std::shared_ptr<ObserverModelAbstract> > get_running_phase_models(
      const std::size_t phase_idx) const;

  /** @brief Return the observer data in one phase */
  std::vector<std::shared_ptr<ActionDataAbstract> > get_running_phase_datas(
      const std::size_t phase_idx) const;

  /** @brief Return the shared parameter managers, one per phase */
  const std::vector<std::shared_ptr<ParameterManager> >& get_params() const;

  /** @brief Return the owned parameter data, one per phase */
  const std::vector<std::shared_ptr<ParameterDataManager> >& get_params_data()
      const;

  /** @brief Return the inclusive running-node start of every phase */
  virtual const std::vector<std::size_t>& get_phase_idxs() const override;

  /** @brief Return the exclusive running-node end of every phase */
  virtual const std::vector<std::size_t>& get_phase_edxs() const override;

  /** @brief Return the optional parameter-constraint managers */
  virtual const std::vector<std::shared_ptr<ConstraintModelManager> >&
  get_parameter_constraints_models() const override;

  /** @brief Return the parameter-constraint data sharing phase payloads */
  virtual const std::vector<std::shared_ptr<ConstraintDataManager> >&
  get_parameter_constraints_datas() const override;

  /** @brief Return true if any phase has active parameter constraints */
  virtual bool has_parameter_constraints() const override;

 private:
  void init(
      const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
      const std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >&
          model_phases,
      std::shared_ptr<ObserverModelAbstract> terminal_model,
      const std::vector<std::shared_ptr<ConstraintModelManager> >&
          parameter_constraints);

  Scalar cost_;
  std::size_t T_;
  VectorXs x0_;
  std::size_t nx_;
  std::size_t ndx_;
  bool is_updated_;
  std::shared_ptr<ActionModelAbstract> terminal_model_;
  std::shared_ptr<ActionDataAbstract> terminal_data_;
  std::vector<std::shared_ptr<ActionModelAbstract> > running_models_;
  std::vector<std::shared_ptr<ActionDataAbstract> > running_datas_;
  std::size_t n_phases_;
  std::vector<std::shared_ptr<ParameterManager> > params_;
  std::vector<std::shared_ptr<ParameterDataManager> > params_data_;
  std::vector<std::shared_ptr<ConstraintModelManager> > parameter_constraints_;
  std::vector<std::shared_ptr<ConstraintDataManager> >
      parameter_constraints_data_;
  std::vector<std::size_t> phase_start_;
  std::vector<std::size_t> phase_end_;
};

}  // namespace crocoddyl

#include "crocoddyl/core/optctrl/observation.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ObservationProblemTpl)

#endif  // CROCODDYL_CORE_OPTCTRL_OBSERVATION_HPP_
