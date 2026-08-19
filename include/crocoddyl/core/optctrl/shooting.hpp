///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files. All
// rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
#define CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/optctrl/problem-abstract.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief This class encapsulates a shooting problem
 *
 * A shooting problem encapsulates the initial state
 * \f$\mathbf{x}_{0}\in\mathcal{M}\f$, a set of running action models and a
 * terminal action model for a discretized trajectory into \f$T\f$ nodes. It has
 * three main methods - `calc`, `calcDiff` and `rollout`. The first computes the
 * set of next states and cost values per each node \f$k\f$. Instead, `calcDiff`
 * updates the derivatives of all action models. Finally, `rollout` integrates
 * the system dynamics. This class is used to decouple problem formulation and
 * resolution.
 *
 * A shooting problem can optionally own action parameters shared by one or
 * more phases. All action data in a parameter phase refer to the same
 * `ParameterDataManagerTpl`, and the terminal node uses the final phase.
 * Parameter layouts are fixed at construction. Structural mutation is
 * supported only for problems without phase-owned data; parameterized problems
 * must be reconstructed when their model or phase layout changes.
 */
template <typename _Scalar>
class ShootingProblemTpl : public ProblemAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionModelAbstractTpl<Scalar> ActionModelAbstract;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the shooting problem and allocate its data
   *
   * @param[in] x0              Initial state
   * @param[in] running_models  Running action models (size \f$T\f$)
   * @param[in] terminal_model  Terminal action model
   */
  ShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
      std::shared_ptr<ActionModelAbstract> terminal_model);

  /**
   * @brief Initialize the shooting problem (models and datas)
   *
   * @param[in] x0              Initial state
   * @param[in] running_models  Running action models (size \f$T\f$)
   * @param[in] terminal_model  Terminal action model
   * @param[in] running_datas   Running action datas (size \f$T\f$)
   * @param[in] terminal_data   Terminal action data
   */
  ShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      const std::vector<std::shared_ptr<ActionDataAbstract> >& running_datas,
      std::shared_ptr<ActionDataAbstract> terminal_data);

  /**
   * @brief Construct a single-phase parameterized shooting problem
   *
   * @param[in] x0 Initial state
   * @param[in] running_models Running action models
   * @param[in] terminal_model Terminal action model
   * @param[in] params_model Parameter manager shared by all nodes
   */
  ShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      std::shared_ptr<ParameterManager> params_model);

  /** @brief Construct a constrained single-phase parameterized problem */
  ShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      std::shared_ptr<ParameterManager> params_model,
      std::shared_ptr<ConstraintModelManager> params_constraint_model);

  /**
   * @brief Construct a multi-phase parameterized shooting problem
   *
   * @param[in] x0 Initial state
   * @param[in] model_phases Running action models grouped by parameter phase
   * @param[in] terminal_model Terminal action model
   * @param[in] params_model One parameter manager per phase
   */
  ShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
          model_phases,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      const std::vector<std::shared_ptr<ParameterManager> >& params_model);

  /** @brief Construct a constrained multi-phase parameterized problem */
  ShootingProblemTpl(
      const VectorXs& x0,
      const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
          model_phases,
      std::shared_ptr<ActionModelAbstract> terminal_model,
      const std::vector<std::shared_ptr<ParameterManager> >& params_model,
      const std::vector<std::shared_ptr<ConstraintModelManager> >&
          params_constraint_model);

  /**
   * @brief Initialize the shooting problem
   */
  ShootingProblemTpl(const ShootingProblemTpl<Scalar>& problem);
  ~ShootingProblemTpl();

  /**
   * @brief Compute the cost and the next states
   *
   * For each node \f$k\f$, and along the state \f$\mathbf{x_{s}}\f$ and control
   * \f$\mathbf{u_{s}}\f$ trajectory, it computes the next state
   * \f$\mathbf{x}_{k+1}\f$ and cost \f$l_{k}\f$.
   *
   * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size
   * \f$T+1\f$)
   * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size
   * \f$T\f$)
   * @return The total cost value \f$l_{k}\f$
   */
  virtual Scalar calc(const std::vector<VectorXs>& xs,
                      const std::vector<VectorXs>& us) override;

  /**
   * @brief Compute the derivatives of the cost and dynamics
   *
   * For each node \f$k\f$, and along the state \f$\mathbf{x_{s}}\f$ and control
   * \f$\mathbf{u_{s}}\f$ trajectory, it computes the derivatives of the cost
   * \f$(\mathbf{l}_{\mathbf{x}}, \mathbf{l}_{\mathbf{u}},
   * \mathbf{l}_{\mathbf{xx}}, \mathbf{l}_{\mathbf{xu}},
   * \mathbf{l}_{\mathbf{uu}})\f$ and dynamics \f$(\mathbf{f}_{\mathbf{x}},
   * \mathbf{f}_{\mathbf{u}})\f$.
   *
   * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size
   * \f$T+1\f$)
   * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size
   * \f$T\f$)
   * @return The total cost value \f$l_{k}\f$
   */
  virtual Scalar calcDiff(const std::vector<VectorXs>& xs,
                          const std::vector<VectorXs>& us) override;

  /**
   * @brief Integrate the dynamics given a control sequence
   *
   * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size
   * \f$T+1\f$)
   * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size
   * \f$T\f$)
   */
  virtual void rollout(const std::vector<VectorXs>& us,
                       std::vector<VectorXs>& xs) override;

  /**
   * @copybrief rollout
   *
   * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size
   * \f$T\f$)
   * @return the time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size
   * \f$T+1\f$)
   */
  virtual std::vector<VectorXs> rollout_us(
      const std::vector<VectorXs>& us) override;

  /**
   * @brief Compute the quasic static commands given a state trajectory
   *
   * @param[out] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size
   * \f$T\f$)
   * @param[in]  xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size
   * \f$T+1\f$)
   */
  void quasiStatic(std::vector<VectorXs>& us, const std::vector<VectorXs>& xs);

  /**
   * @copybrief quasiStatic
   *
   * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size
   * \f$T+1\f$)
   * @return the time-discrete quasic static commands \f$\mathbf{u_{s}}\f$ (size
   * \f$T\f$)
   */
  std::vector<VectorXs> quasiStatic_xs(const std::vector<VectorXs>& xs);

  /**
   * @brief Circular append of the model and data onto the end running node
   *
   * Once we update the end running node, the first running mode is removed as
   * in a circular buffer.
   *
   * @param[in] model  action model
   * @param[in] data   action data
   */
  void circularAppend(std::shared_ptr<ActionModelAbstract> model,
                      std::shared_ptr<ActionDataAbstract> data);

  /**
   * @copybrief circularAppend
   *
   * Once we update the end running node, the first running mode is removed as
   * in a circular buffer. Note that this method allocates new data for the end
   * running node.
   *
   * @param[in] model  action model
   */
  void circularAppend(std::shared_ptr<ActionModelAbstract> model);

  /**
   * @brief Update the model and data for a specific node
   *
   * @param[in] i      node index \f$(0\leq i < T+1)\f$
   * @param[in] model  action model
   * @param[in] data   action data
   */
  void updateNode(const std::size_t i,
                  std::shared_ptr<ActionModelAbstract> model,
                  std::shared_ptr<ActionDataAbstract> data);

  /**
   * @brief Update a model and allocated new data for a specific node
   *
   * @param[in] i      node index \f$(0\leq i < T+1)\f$
   * @param[in] model  action model
   */
  void updateModel(const std::size_t i,
                   std::shared_ptr<ActionModelAbstract> model);

  /**
   * @brief Cast the shooting problem to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ShootingProblemTpl<NewScalar> A shooting problem with the
   * new scalar type.
   */
  template <typename NewScalar>
  ShootingProblemTpl<NewScalar> cast() const;

  /**
   * @brief Return the number of running nodes
   */
  virtual std::size_t get_T() const override;

  /**
   * @brief Return the initial state
   */
  virtual const VectorXs& get_x0() const override;

  /** @brief Return all running models flattened in time order */
  virtual const std::vector<std::shared_ptr<ActionModelAbstract> >&
  get_runningModels() const override;

  /**
   * @brief Return the terminal model
   */
  virtual const std::shared_ptr<ActionModelAbstract>& get_terminalModel()
      const override;

  /** @brief Return all running data flattened in time order */
  virtual const std::vector<std::shared_ptr<ActionDataAbstract> >&
  get_runningDatas() const override;

  /**
   * @brief Return the terminal data
   */
  virtual const std::shared_ptr<ActionDataAbstract>& get_terminalData()
      const override;

  /**
   * @brief Modify the initial state
   */
  void set_x0(const VectorXs& x0_in);

  /**
   * @brief Modify the running models and allocate new data
   */
  void set_runningModels(
      const std::vector<std::shared_ptr<ActionModelAbstract> >& models);

  /**
   * @brief Modify the terminal model and allocate new data
   */
  void set_terminalModel(std::shared_ptr<ActionModelAbstract> model);

  /**
   * @brief Modify the number of threads using with multithreading support
   *
   * For values lower than 1, the number of threads is chosen by
   * CROCODDYL_WITH_NTHREADS macro
   */
  void set_nthreads(const int nthreads);

  /**
   * @brief Modify the is_updated flag
   */
  virtual void set_is_updated(const bool is_updated) override;

  /**
   * @brief Return the dimension of the state tuple
   */
  virtual std::size_t get_nx() const override;

  /**
   * @brief Return the dimension of the tangent space of the state manifold
   */
  virtual std::size_t get_ndx() const override;

  /**
   * @brief Return the number of threads
   */
  virtual std::size_t get_nthreads() const override;

  /**
   * @brief Return only once true is the shooting problem has been changed,
   * otherwise false
   */
  virtual bool is_updated() override;

  /** @brief Update the active parameter vector of one phase exactly once */
  virtual void update_p(const Eigen::Ref<const VectorXs>& p,
                        const std::size_t phase_idx = 0) override;

  /** @brief Return the number of parameterized running phases */
  virtual std::size_t get_n_phases() const override;

  /**
   * @brief Return the running action models of a parameter phase
   *
   * @param[in] phase_idx Index of the parameter phase
   */
  std::vector<std::shared_ptr<ActionModelAbstract> > get_runningPhaseModels(
      const std::size_t phase_idx) const;

  /**
   * @brief Return the running action data of a parameter phase
   *
   * @param[in] phase_idx Index of the parameter phase
   */
  std::vector<std::shared_ptr<ActionDataAbstract> > get_runningPhaseDatas(
      const std::size_t phase_idx) const;

  /** @brief Return the shared parameter models, one per phase */
  const std::vector<std::shared_ptr<ParameterManager> >& get_paramsModel()
      const;

  /** @brief Return the owned parameter data, one per phase */
  const std::vector<std::shared_ptr<ParameterDataManager> >& get_paramsData()
      const;

  /** @brief Return the inclusive running-node start of every parameter phase */
  virtual const std::vector<std::size_t>& get_phase_idxs() const override;

  /** @brief Return the exclusive running-node end of every parameter phase */
  virtual const std::vector<std::size_t>& get_phase_edxs() const override;

  /** @brief Return the optional parameter-constraint managers */
  virtual const std::vector<std::shared_ptr<ConstraintModelManager> >&
  get_paramsConstraintModel() const override;

  /** @brief Return the parameter-constraint data sharing phase payloads */
  virtual const std::vector<std::shared_ptr<ConstraintDataManager> >&
  get_paramsConstraintData() const override;

  /** @brief Return true if any phase has active parameter constraints */
  virtual bool has_parameter_constraints() const override;

  /**
   * @brief Print information on the 'ShootingProblem'
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const ShootingProblemTpl<Scalar>& problem);

 protected:
  Scalar cost_;    //!< Total cost
  std::size_t T_;  //!< number of running nodes
  VectorXs x0_;    //!< Initial state
  std::shared_ptr<ActionModelAbstract>
      terminal_model_;  //!< Terminal action model
  std::shared_ptr<ActionDataAbstract> terminal_data_;  //!< Terminal action data
  std::vector<std::shared_ptr<ActionModelAbstract> >
      running_models_;  //!< Running action model
  std::vector<std::shared_ptr<ActionDataAbstract> >
      running_datas_;     //!< Running action data
  std::size_t nx_;        //!< State dimension
  std::size_t ndx_;       //!< State rate dimension
  std::size_t nthreads_;  //!< Number of threads launch by the multi-threading
                          //!< application
  bool is_updated_;

  std::size_t n_phases_;  //!< Number of parameter phases (zero if disabled)
  std::vector<std::shared_ptr<ParameterManager> > params_model_;
  std::vector<std::shared_ptr<ParameterDataManager> > params_data_;
  std::vector<std::shared_ptr<ConstraintModelManager> >
      params_constraint_model_;
  std::vector<std::shared_ptr<ConstraintDataManager> > params_constraint_data_;
  std::vector<std::size_t> phase_start_;
  std::vector<std::size_t> phase_end_;

 private:
  void allocateData();

  static std::shared_ptr<ActionModelAbstract> checkedTerminalModel(
      std::shared_ptr<ActionModelAbstract> terminal_model);

  static std::vector<std::shared_ptr<ActionModelAbstract> > flattenModelPhases(
      const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
          model_phases);

  void initParameterization(
      const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
          model_phases,
      const std::vector<std::shared_ptr<ConstraintModelManager> >&
          params_constraint_model);
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/optctrl/shooting.hxx"

extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::ShootingProblemTpl<double>;
extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::ShootingProblemTpl<float>;

#endif  // CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
