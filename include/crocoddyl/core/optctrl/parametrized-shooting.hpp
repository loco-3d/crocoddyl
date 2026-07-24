///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OPTCTRL_PARAMETRIZED_SHOOTING_HPP_
#define CROCODDYL_CORE_OPTCTRL_PARAMETRIZED_SHOOTING_HPP_

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief Shooting problem with phase-shared action parameters
 *
 * This problem preserves the trajectory semantics of `ShootingProblemTpl`.
 * Each non-empty running phase owns one shared `ParameterManagerTpl` and one
 * `ParameterDataManagerTpl`; all action data in that phase refer to that
 * payload. The terminal node uses the final phase. Phase status/layout changes
 * invalidate existing data, so they must be completed before construction or
 * followed by reconstruction of the problem. The inherited canonical
 * structural-mutation API rejects phased problems before mutation.
 * Initial-state, threading, evaluation, rollout and update-tracking operations
 * remain available.
 */
template <typename _Scalar>
class ParametrizedShootingProblemTpl : public ShootingProblemTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ShootingProblemTpl<Scalar> Base;
  typedef typename Base::ActionModelAbstract ActionModelAbstract;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef typename Base::VectorXs VectorXs;

  /**
   * @brief Construct a single-phase parameterized shooting problem
   *
   * @param[in] x0 Initial state
   * @param[in] running_models Running action models
   * @param[in] terminal_model Terminal action model
   * @param[in] params Parameter manager shared by all nodes
   */
  ParametrizedShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      std::shared_ptr<ParameterManager> params);

  /**
   * @brief Construct a constrained single-phase parameterized problem
   */
  ParametrizedShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      std::shared_ptr<ParameterManager> params,
      std::shared_ptr<ConstraintModelManager> parameter_constraints);

  /**
   * @brief Construct a multi-phase parameterized shooting problem
   *
   * @param[in] x0 Initial state
   * @param[in] model_phases Running action models grouped by phase
   * @param[in] terminal_model Terminal action model
   * @param[in] params One parameter manager per phase
   */
  ParametrizedShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
          model_phases,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      const std::vector<std::shared_ptr<ParameterManager> >& params);

  /**
   * @brief Construct a constrained multi-phase parameterized problem
   */
  ParametrizedShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
          model_phases,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      const std::vector<std::shared_ptr<ParameterManager> >& params,
      const std::vector<std::shared_ptr<ConstraintModelManager> >&
          parameter_constraints);

  virtual ~ParametrizedShootingProblemTpl() = default;

  /** @brief Update the active parameter vector of one phase exactly once */
  virtual void update_p(const Eigen::Ref<const VectorXs>& p,
                        const std::size_t phase_idx = 0) override;

  /** @brief Return the number of parameterized running phases */
  virtual std::size_t get_n_phases() const override;

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

  /** @brief Return the running action models in one phase */
  std::vector<std::shared_ptr<ActionModelAbstract> > get_running_phase_models(
      const std::size_t phase_idx) const;

  /** @brief Return the running action data in one phase */
  std::vector<std::shared_ptr<ActionDataAbstract> > get_running_phase_datas(
      const std::size_t phase_idx) const;

 private:
  static std::shared_ptr<ActionModelAbstract> checkedTerminalModel(
      std::shared_ptr<ActionModelAbstract> terminal_model);

  static std::vector<std::shared_ptr<ActionModelAbstract> > flattenModelPhases(
      const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
          model_phases);

  void init(
      const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
          model_phases,
      const std::vector<std::shared_ptr<ConstraintModelManager> >&
          parameter_constraints);

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

#include "crocoddyl/core/optctrl/parametrized-shooting.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ParametrizedShootingProblemTpl)

#endif  // CROCODDYL_CORE_OPTCTRL_PARAMETRIZED_SHOOTING_HPP_
