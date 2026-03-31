///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_FDDP_HPP_
#define CROCODDYL_CORE_SOLVERS_FDDP_HPP_

#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

enum DynamicsSolverType {
  FeasShoot = 0,  //!< Feasibility-driven DDP
  MultiShoot,     //!< Classical multiple shooting but with Riccati solver
  HybridShoot,    //!< Feasibility-driven multiple rollouts
  SingleShoot     //!< Similar to classical DDP but with xs warmstart
};

enum EqualitySolverType {
  LuNull = 0,  //!< Nullspace factorization using LU decomposition
  QrNull,      //!< Nullspace factorization using QR decomposition
  Schur,       //!< Schur-complement factorization
};

/**
 * @brief Feasibility-driven Differential Dynamic Programming (FDDP) solver
 *
 * The FDDP solver computes an optimal trajectory and control commands by
 * iterates running `backwardPass()` and `forwardPass()`. The backward pass
 * accepts infeasible guess as described in the `SolverFDDP::backwardPass()`.
 * Additionally, the forward pass handles infeasibility simulations that
 * resembles the numerical behaviour of a multiple-shooting formulation, i.e.:
 * \f{eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0 - (1 -
 * \alpha)\mathbf{\bar{f}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
 * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\ \mathbf{\hat{x}}_{k+1} &=&
 * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k) - (1 -
 * \alpha)\mathbf{\bar{f}}_{k+1}.
 * \f}
 * Note that the forward pass keeps the gaps \f$\mathbf{\bar{f}}_s\f$ open
 * according to the step length \f$\alpha\f$ that has been accepted. This solver
 * has shown empirically greater globalization strategy. Additionally, the
 * expected improvement computation considers the gaps in the dynamics:
 * \f{equation}
 *   \Delta J(\alpha) = \Delta_1\alpha + \frac{1}{2}\Delta_2\alpha^2,
 * \f}
 * with
 * \f{eqnarray}
 *   \Delta_1 = \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k}
 * +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k} -
 *   V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
 *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k +
 * \mathbf{\bar{f}}_k^\top(2 V_{\mathbf{xx}_k}\mathbf{x}_k
 * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). \f}
 *
 * For more details about the feasibility-driven differential dynamic
 * programming algorithm see: \include mastalli-icra20.bib
 *
 * \sa `SolverAbstract()`, `backwardPass()`, `forwardPass()`, and
 * `expectedImprovement()`.
 */
template <typename _Scalar>
class SolverFDDPTpl : public SolverAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverFDDPTpl)

  typedef _Scalar Scalar;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ShootingProblem::ActionModelAbstract ActionModelAbstract;
  typedef typename ShootingProblem::ActionDataAbstract ActionDataAbstract;
  typedef CallbackAbstractTpl<Scalar> CallbackAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixXsRowMajor MatrixXsRowMajor;
  using SolverAbstract::computeDynamicFeasibility;
  using SolverAbstract::computeEqualityFeasibility;
  using SolverAbstract::computeFeasibility;
  using SolverAbstract::computeInequalityFeasibility;
  using SolverAbstract::resizeData;

  /**
   * @brief Initialize the FDDP solver
   *
   * @param[in] problem      Shooting problem
   * @param[in] dyn_solver   Type of dynamic solver
   * @param[in] term_solver  Type of terminal solver
   */
  explicit SolverFDDPTpl(std::shared_ptr<ShootingProblem> problem,
                         const DynamicsSolverType dyn_solver = FeasShoot,
                         const EqualitySolverType term_solver = LuNull);
  virtual ~SolverFDDPTpl() = default;

  /**
   * @copybrief SolverAbstract::computeDirection
   */
  virtual void computeDirection(const bool recalc = true) override;

  /**
   * @copybrief SolverAbstract::computeCandidate
   */
  virtual void computeCandidate(const Scalar step_length = Scalar(1.)) override;

  /**
   * @brief Perform a forward pass with a predefined step length
   *
   * Our solver supports fourth type of forward passes: feasiblility-driven
   * (FeasShoot), multiple shooting (MultiShoot), hybrid shooting (HybridShoot),
   * and single shooting (SingleShoot). The feasibility-driven shooting
   * decreases monotonocally the dynamics feasibility based on the applied step
   * length. The hybrid shooting combines both feasibility-driven and multiple
   * shooting approaches. Instead, the single shooting is the traditional
   * approach implemented in DDP. The type of dynamics solver can be defined via
   * `set_dynamics_solver`.
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   */
  void forwardPass(const Scalar steplength = Scalar(1.));

  /**
   * @brief Update the dual and slack variables with a predefined step length
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   */
  virtual void updateDualsAndSlacks(const Scalar stepLength = Scalar(1.));

  /**
   * @copybrief SolverAbstract::stoppingCriteria
   */
  virtual Scalar stoppingCriteria() override;

  /**
   * @copybrief SolverAbstract::expectedImprovement
   *
   * The expected improvement computation considers the gaps in the dynamics:
   * \f{equation} \Delta J(\alpha) = \Delta_1\alpha +
   * \frac{1}{2}\Delta_2\alpha^2, \f} with \f{eqnarray} \Delta_1 =
   * \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k}
   * +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k}
   * - V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
   *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k +
   * \mathbf{\bar{f}}_k^\top(2 V_{\mathbf{xx}_k}\mathbf{x}_k
   * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). \f}
   */
  virtual Vector3s expectedImprovement() override;

  /**
   * @copybrief SolverAbstract::computeMeritFunctionImprovement
   */
  virtual void computeMeritFunctionImprovement() override;

  /**
   * @copybrief SolverAbstract::computeExpectedMeritFunctionImprovement
   */
  virtual void computeExpectedMeritFunctionImprovement() override;

  /**
   * @brief Update the merit function value for the current guess
   */
  virtual void updateMeritFunction() override;

  /**
   * @brief Check if we should accept or not the step
   *
   * @return True if we should accept the step. False otherwise
   */
  virtual bool checkAcceptance() override;

  /**
   * @brief Run the backward pass (Riccati sweep)
   *
   * It assumes that the Jacobian and Hessians of the optimal control problem
   * have been compute (i.e., `calcDir()`). The backward pass handles
   * infeasible guess through a modified Riccati sweep: \f{eqnarray*}
   *   \mathbf{Q}_{\mathbf{x}_k} &=& \mathbf{l}_{\mathbf{x}_k} +
   * \mathbf{f}^\top_{\mathbf{x}_k} (V_{\mathbf{x}_{k+1}}
   * +
   * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
   *   \mathbf{Q}_{\mathbf{u}_k} &=& \mathbf{l}_{\mathbf{u}_k} +
   * \mathbf{f}^\top_{\mathbf{u}_k} (V_{\mathbf{x}_{k+1}}
   * +
   * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
   *   \mathbf{Q}_{\mathbf{xx}_k} &=& \mathbf{l}_{\mathbf{xx}_k} +
   * \mathbf{f}^\top_{\mathbf{x}_k} V_{\mathbf{xx}_{k+1}}
   * \mathbf{f}_{\mathbf{x}_k},\\
   *   \mathbf{Q}_{\mathbf{xu}_k} &=& \mathbf{l}_{\mathbf{xu}_k} +
   * \mathbf{f}^\top_{\mathbf{x}_k} V_{\mathbf{xx}_{k+1}}
   * \mathbf{f}_{\mathbf{u}_k},\\
   *   \mathbf{Q}_{\mathbf{uu}_k} &=& \mathbf{l}_{\mathbf{uu}_k} +
   * \mathbf{f}^\top_{\mathbf{u}_k} V_{\mathbf{xx}_{k+1}}
   * \mathbf{f}_{\mathbf{u}_k}, \f} where
   * \f$\mathbf{l}_{\mathbf{x}_k}\f$,\f$\mathbf{l}_{\mathbf{u}_k}\f$,\f$\mathbf{f}_{\mathbf{x}_k}\f$
   * and \f$\mathbf{f}_{\mathbf{u}_k}\f$ are the Jacobians of the cost function
   * and dynamics,
   * \f$\mathbf{l}_{\mathbf{xx}_k}\f$,\f$\mathbf{l}_{\mathbf{xu}_k}\f$ and
   * \f$\mathbf{l}_{\mathbf{uu}_k}\f$ are the Hessians of the cost function,
   * \f$V_{\mathbf{x}_{k+1}}\f$ and \f$V_{\mathbf{xx}_{k+1}}\f$ defines the
   * linear-quadratic approximation of the Value function, and
   * \f$\mathbf{\bar{f}}_{k+1}\f$ describes the gaps of the dynamics.
   */
  virtual void backwardPass();

  /**
   * @brief Run the batch pass to account for its constraints
   *
   * Update the direction and feed-forward term to account for the terminal
   * constraint. To do so, we first compute the unscaled search direction
   * accounting for the terminal constraint.
   */
  void batchPass();

  /**
   * @brief Update search direction associated with the batch's constraints
   */
  void updateDir();

  /**
   * @brief Compute the linear-quadratic approximation of the control
   * action-value function
   *
   * @param[in] t      Time instance
   * @param[in] model  Action model in the given time instance
   * @param[in] data   Action data in the given time instance
   */
  virtual void computeActionValueFunction(
      const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model,
      const std::shared_ptr<ActionDataAbstract>& data);

  /**
   * @brief Compute the linear-quadratic approximation of the control
   * action-value function associated to the batch's constraints
   *
   * @param[in] t      Time instance
   * @param[in] data   Action data in the given time instance
   */
  virtual void computeBatchActionValueFunction(
      const std::size_t t, const std::shared_ptr<ActionDataAbstract>& data);

  /**
   * @brief Compute the feedforward and feedback terms (control policy) computed
   * via a Cholesky decomposition
   *
   * To compute the feedforward \f$\mathbf{k}_k\f$ and feedback
   * \f$\mathbf{K}_k\f$ terms, we use a Cholesky decomposition to solve
   * \f$\mathbf{Q}_{\mathbf{uu}_k}^{-1}\f$ term: \f{eqnarray}
   * \mathbf{k}_k &=& \mathbf{Q}_{\mathbf{uu}_k}^{-1}\mathbf{Q}_{\mathbf{u}},\\
   * \mathbf{K}_k &=& \mathbf{Q}_{\mathbf{uu}_k}^{-1}\mathbf{Q}_{\mathbf{ux}}.
   * \f}
   *
   * Note that if the Cholesky decomposition fails, then we re-start the
   * backward pass and increase the state and control regularization values.
   */
  virtual void computePolicy(const std::size_t t);

  /**
   * @brief Compute the feedforward and feedback terms (control policy)
   * associated to the batch's constraints.
   *
   * Note that if the Cholesky decomposition fails, then we re-start the
   * backward pass and increase the state and control regularization values.
   */
  virtual void computeBatchPolicy(const std::size_t t);

  /**
   * @brief Compute the linear-quadratic approximation of the value function
   *
   * This function is called in the backward pass after updating the local
   * action-value and policy functions.
   *
   * @param[in] t      Time instance
   * @param[in] model  Action model in the given time instance
   */
  virtual void computeValueFunction(
      const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model);

  /**
   * @brief Compute the linear-quadratic approximation of the value function
   * associated to the batch's constraints
   *
   * This function is called in the backward pass after updating the local
   * action-value and policy functions.
   *
   * @param[in] t  Time instance
   */
  virtual void computeBatchValueFunction(const std::size_t t);

  /**
   * @brief Perform a linear rollout give the current control policy
   *
   * The results of this linear rollout are stored in dxs and dus.
   */
  void linearRollout();

  /**
   * @brief Run the feasibility-driven nonlinear rollout
   *
   * It rollouts the action model given the feasibility-driven approach
   * described in "Crocoddyl: An Efficient and Versatile Framework for
   * Multi-Contact Optimal Control"
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   */
  virtual void feasShootForwardPass(const Scalar steplength);

  /**
   * @brief Run the multiple-shooting rollout
   *
   * It rollouts the action model given the multiple-shooting approach described
   * in (TODO: add paper title)
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   */
  virtual void multiShootForwardPass(const Scalar steplength);

  /**
   * @brief Run the multiple-shooting rollout with intervals of
   * feasibility-driven search
   *
   * It rollouts the action model given the hybrid-shooting approach described
   * in (TODO: add paper title)
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   */
  virtual void hybridShootForwardPass(const Scalar steplength);

  /**
   * @brief Run the classical nonlinear rollout
   *
   * It rollouts the action model given the classical approach in DDP. You can
   * find details in "A second-order gradient method for determining optimal
   * trajectories of non-linear discrete-time systems"
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   */
  virtual void singleShootForwardPass(const Scalar steplength);

  /**
   * @brief Update the candidate solution: cost, feasibilities, and merit value
   */
  void updateCandidate() override;

  /**
   * @brief Criteria used to decrease regularization
   */
  bool decreaseRegularizationCriteria() override;

  /**
   * @brief Criteria used to increase regularization
   */
  bool increaseRegularizationCriteria() override;

  /**
   * @brief Increase the state and control regularization values by a
   * `regfactor_` factor
   */
  void increaseRegularization() override;

  /**
   * @brief Decrease the state and control regularization values by a
   * `regfactor_` factor
   */
  void decreaseRegularization() override;

  /**
   * @brief Cast the FDDP solver to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return SolverFDDPTpl<NewScalar> A FDDP solver with the new scalar type.
   */
  template <typename NewScalar>
  SolverFDDPTpl<NewScalar> cast() const;

  /**
   * @brief Return the type of solver used for handling the dynamics constraints
   */
  DynamicsSolverType get_dynamics_solver() const;

  /**
   * @brief Return the type of solver used for handling the terminal constraints
   */
  EqualitySolverType get_terminal_solver() const;

  /**
   * @brief Return the regularization factor used to increase the damping value
   */
  Scalar get_reg_incfactor() const;

  /**
   * @brief Return the regularization factor used to decrease the damping value
   */
  Scalar get_reg_decfactor() const;

  /**
   * @brief Return the tolerance of the expected gradient used for testing the
   * step
   */
  Scalar get_th_grad() const;

  /**
   * @brief Return the step-length threshold used to decrease regularization
   */
  Scalar get_th_stepdec() const;

  /**
   * @brief Return the step-length threshold used to increase regularization
   */
  Scalar get_th_stepinc() const;

  /**
   * @brief Return the minimum improvement threshold used to increase
   * regularization
   */
  Scalar get_th_minimprove() const;

  /**
   * @brief Return the threshold used for accepting step along ascent direction
   */
  Scalar get_th_acceptnegstep() const;

  /**
   * @brief Return the threshold used for accepting minimum steps
   */
  Scalar get_th_acceptminstep() const;

  /**
   * @brief Return the rho parameter used in the merit function
   */
  Scalar get_rho() const;

  /**
   * @brief Return the threshold for switching to feasibility
   */
  Scalar get_th_minfeas() const;

  /**
   * @brief Return the estimated penalty parameter that balances relative
   * contribution of the cost function and equality constraints
   */
  Scalar get_upsilon() const;

  /**
   * @brief Return the upsilon decresing factor used to estimate to balance
   * optimality and feasibility
   */
  Scalar get_upsilon_decfactor() const;

  /**
   * @brief Return the zero-upsilon label
   *
   * True if we set the estimated penalty parameter (upsilon) to zero when solve
   * is called.
   */
  bool get_zero_upsilon() const;

  /**
   * @brief Return the hybrid shooting intervals
   */
  const std::vector<std::size_t>& get_Ts() const;

  /**
   * @brief Return the Hessian of the Value function \f$V_{\mathbf{xx}_s}\f$
   */
  const std::vector<MatrixXs>& get_Vxx() const;

  /**
   * @brief Return the Hessian of the Value function \f$V_{\mathbf{x}_s}\f$
   */
  const std::vector<VectorXs>& get_Vx() const;

  /**
   * @brief Return the Hessian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{xx}_s}\f$
   */
  const std::vector<MatrixXs>& get_Qxx() const;

  /**
   * @brief Return the Hessian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{xu}_s}\f$
   */
  const std::vector<MatrixXs>& get_Qxu() const;

  /**
   * @brief Return the Hessian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{uu}_s}\f$
   */
  const std::vector<MatrixXs>& get_Quu() const;

  /**
   * @brief Return the Jacobian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{x}_s}\f$
   */
  const std::vector<VectorXs>& get_Qx() const;

  /**
   * @brief Return the Jacobian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{u}_s}\f$
   */
  const std::vector<VectorXs>& get_Qu() const;

  /**
   * @brief Return the feedback gains \f$\mathbf{K}_{s}\f$
   */
  const std::vector<MatrixXsRowMajor>& get_K() const;

  /**
   * @brief Return the feedforward gains \f$\mathbf{k}_{s}\f$
   */
  const std::vector<VectorXs>& get_k() const;

  /**
   * @brief Return the Hessian of the Value function \f$V_{\mathbf{xc}_s}\f$
   */
  const std::vector<MatrixXs>& get_Vxc() const;

  /**
   * @brief Return the Hessian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{xc}_s}\f$
   */
  const std::vector<MatrixXs>& get_Qxc() const;

  /**
   * @brief Return the Hessian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{uc}_s}\f$
   */
  const std::vector<MatrixXs>& get_Quc() const;

  /**
   * @brief Return the linear update in \f$\delta\mathbf{X}_{c_s}\f$
   */
  const std::vector<MatrixXs>& get_dXc() const;

  /**
   * @brief Return the linear update in \f$\delta\mathbf{U}_{c_s}\f$
   */
  const std::vector<MatrixXs>& get_dUc() const;

  /**
   * @brief Return the feedforward gains \f$\mathbf{K}_{c_s}\f$
   */
  const std::vector<MatrixXs>& get_Kc() const;

  /**
   * @brief Return the Jacobian of the terminal constraint gains
   * \f$\delta\mathbf{H}_{c}\f$
   */
  const MatrixXs& get_dHc() const;

  /**
   * @brief Return the bias term of the terminal constraint gains
   * \f$\mathbf{h}_{c}\f$
   */
  const VectorXs& get_hc() const;

  /**
   * @brief Return the next terminal-constraint multiplier
   * \f$\boldsymbol{\beta}^+\f$
   */
  const VectorXs& get_beta_plus() const;

  /**
   * @brief Set the dynamic solver used for handling the dynamics constraints
   *
   * It is worth noting that the default solver is the Feasibility-Driven DDP.
   * When we enable parallelization, this strategy is not necessarily the faster
   * one for medium to large systems.
   *
   * @param[in] type  Type of dynamics solver
   * @param[in] Tshoot  Number of nodes per each shooting interval
   */
  void set_dynamics_solver(const DynamicsSolverType type,
                           const std::size_t Tshoot = 0);

  /**
   * @brief Set the type of solver used for handling the terminal constraints
   *
   * @param[in] type  Type of terminal solver
   */
  void set_terminal_solver(const EqualitySolverType type);

  /**
   * @brief Modify the regularization factor used to increase the damping value
   */
  void set_reg_incfactor(const Scalar reg_factor);

  /**
   * @brief Modify the regularization factor used to decrease the damping value
   */
  void set_reg_decfactor(const Scalar reg_factor);

  /**
   * @brief Modify the tolerance of the expected gradient used for testing the
   * step
   */
  void set_th_grad(const Scalar th_grad);

  /**
   * @brief Modify the threshold used to accept steps that cannot be be improved
   * due to numerical errors the th noimprovement object
   */
  void set_th_noimprovement(const Scalar th_noimprovement);

  /**
   * @brief Modify the step-length threshold used to decrease regularization
   */
  void set_th_stepdec(const Scalar th_step);

  /**
   * @brief Modify the step-length threshold used to increase regularization
   */
  void set_th_stepinc(const Scalar th_step);

  /**
   * @brief Modify the minimum improvement threshold used to increase
   * regularization
   */
  void set_th_minimprove(const Scalar th_step);

  /**
   * @brief Modify the threshold used for accepting step along ascent direction
   */
  void set_th_acceptnegstep(const Scalar th_acceptnegstep);

  /**
   * @brief Modify the threshold used for accepting minimum steps
   */
  void set_th_acceptminstep(const Scalar th_acceptminstep);

  /**
   * @brief Modify the rho parameter used in the merit function
   */
  void set_rho(const Scalar rho);

  /**
   * @brief Modify the threshold for switching to feasibility
   */
  void set_th_minfeas(const Scalar th_minfeas);

  /**
   * @brief Modify the upsilon decresing factor used to estimate to balance
   * optimality and feasibility
   */
  void set_upsilon_decfactor(const Scalar th_step);

  /**
   * @brief Modify the zero-upsilon label
   *
   * @param zero_upsilon  True if we set estimated penalty parameter (upsilon)
   * to zero when solve is called.
   */
  void set_zero_upsilon(const bool zero_upsilon);

 protected:
  /**
   * @brief Allocate all the internal data needed for the solver
   */
  void allocateData();

  /**
   * @copybrief SolverAbstract::resizeRunningData
   */
  virtual void resizeRunningData() override;

  /**
   * @copybrief SolverAbstract::resizeTerminalData
   */
  virtual void resizeTerminalData() override;

  /**
   * @brief compute the multiplier associated to the terminal constraint
   */
  void computeNullTerminalMultiplier();

  DynamicsSolverType dyn_solver_;   //!< Type of dynamics solver
  EqualitySolverType term_solver_;  //!< Type of terminal solver
  Scalar reg_incfactor_;  //!< Regularization factor used to increase the
                          //!< damping value
  Scalar reg_decfactor_;  //!< Regularization factor used to decrease the
                          //!< damping value
  Scalar th_grad_;  //!< Tolerance of the expected gradient used for testing the
                    //!< step
  Scalar th_noimprovement_;  //!< Threshold used to accept steps that cannot be
                             //!< be improved due to numerical errors
  Scalar
      th_stepdec_;  //!< Step-length threshold used to decrease regularization
  Scalar
      th_stepinc_;  //!< Step-length threshold used to increase regularization
  Scalar th_minimprove_;     //!< Minimum improvement threshold used in the
                             //!< regularization scheme
  Scalar th_acceptnegstep_;  //!< Threshold used for accepting step along ascent
                             //!< direction
  Scalar th_acceptminstep_;  //!< Threshold used for accepting step along with a
                             //!< minimum length
  Scalar rho_;         //!< Parameter used in the merit function to predict the
                       //!< expected reduction
  Scalar th_minfeas_;  //!< Threshold for switching to feasibility
  Scalar
      upsilon_;  //!< Estimated penalty parameter that balances relative
                 //!< contribution of the cost function and equality constraints
  Scalar upsilon_decfactor_;  //!< Estimated penalty parameter factor used to
                              //!< decrease its value
  bool zero_upsilon_;  //!< True if we wish to set estimated penalty parameter
                       //!< (upsilon) to zero when solve is called.
  std::vector<std::size_t> Ts_;  //!< Index that describes the hybrid shoots

  // allocate data
  MatrixXs Vxx_tmp_;  //!< Temporary variable for ensuring symmetry of Vxx
  std::vector<MatrixXs>
      Vxx_;  //!< Hessian of the Value function \f$\mathbf{V_{xx}}\f$
  std::vector<VectorXs>
      Vxx_f_;  //!< Hessian of the Value function times the gap
               //!< \f$\mathbf{V_{xx} \bar{f}}\f$
  std::vector<VectorXs>
      Vx_;  //!< Gradient of the Value function \f$\mathbf{V_x}\f$
  std::vector<VectorXs>
      Lxx_dx_;  //!< Second-order change of the cost function
                //!< \f$\boldsymbol{\ell}_{\mathbf{{xx}}}\delta\mathbf{x}\f$
  std::vector<VectorXs>
      Luu_du_;  //!< Second-order change of the cost function
                //!< \f$\boldsymbol{\ell}_{\mathbf{{uu}}}\delta\mathbf{u}\f$
  std::vector<VectorXs>
      Lxu_du_;  //!< Second-order change of the cost function
                //!< \f$\boldsymbol{\ell}_{\mathbf{{xu}}}\delta\mathbf{u}\f$
  std::vector<MatrixXs>
      Qxx_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{xx}}\f$
  std::vector<MatrixXs>
      Qxu_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{xu}}\f$
  std::vector<MatrixXs>
      Quu_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{uu}}\f$
  std::vector<VectorXs>
      Qx_;  //!< Gradient of the Hamiltonian \f$\mathbf{Q_x}\f$
  std::vector<VectorXs>
      Qu_;  //!< Gradient of the Hamiltonian \f$\mathbf{Q_u}\f$
  std::vector<MatrixXsRowMajor> K_;  //!< Feedback gains \f$\mathbf{K}\f$
  std::vector<VectorXs> k_;          //!< Feed-forward terms \f$\mathbf{l}\f$
  std::vector<VectorXs>
      dx_;  //!< State error during the roll-out/forward-pass (size T)
  std::vector<MatrixXsRowMajor>
      FxTVxx_p_;  //!< Store the value of
                  //!< \f$\mathbf{f_x}^T\mathbf{V_{xx}}^{'}\f$
  std::vector<MatrixXsRowMajor>
      FuTVxx_p_;      //!< Store the values of
                      //!< \f$\mathbf{f_u}^T\mathbf{V_{xx}}^{'}\f$
                      //!< per each running node
  VectorXs fTVxx_p_;  //!< Store the value of
                      //!< \f$\mathbf{\bar{f}}^T\mathbf{V_{xx}}^{'}\f$
  std::vector<Eigen::LLT<MatrixXs> > Quu_llt_;  //!< Cholesky LLT solver
  std::vector<VectorXs>
      Quuk_;  //!< Store the values of \f$\mathbf{Q_{uu}\mathbf{k}} per each
              //!< running node

  std::size_t dHc_rank_;  //!< Rank of the Jacobian of the terminal constraint
  std::vector<MatrixXs>
      Vxc_;  //!< Gradient of the Value function \f$\mathbf{V_{xc}}\f$
             //!< associated to the terminal constraint
  std::vector<MatrixXs>
      Qxc_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{xc}}\f$ associated to
             //!< the terminal constraint
  std::vector<MatrixXs>
      Quc_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{uc}}\f$ associated to
             //!< the terminal constraint
  std::vector<MatrixXs> dXc_;  //!< Linear state direction (size T + 1)
                               //!< associated to the terminal constraint
  std::vector<MatrixXs> dUc_;  //!< Linear control direction (size T)
                               //!< associated to the terminal constraint
  std::vector<MatrixXs> Kc_;   //!< Feed-forward terms \f$\mathbf{K_c}\f$
                               //!< associated to the terminal constraint
  MatrixXs dHc_;               //!< Jacobian of the terminal constraint
  VectorXs hc_;                //!< Bias term of the terminal constraint
  MatrixXs YZc_;
  VectorXs Yhc_;
  MatrixXs dHcY_;
  MatrixXs YdHcY_;
  VectorXs beta_plus_;  //!< Next value of the terminal-constraint multiplier
  Eigen::LLT<MatrixXs>
      YdHcY_llt_;  //!< Cholesky LLT solver for the terminal constraint
  Eigen::FullPivLU<MatrixXs>
      dHc_lu_;  //!< Full-pivot LU solvers used for computing the span and
                //!< nullspace matrices of the terminal constraint
  Eigen::ColPivHouseholderQR<MatrixXs>
      dHc_qr_;  //!< Column-pivot QR solvers used for computing the span and
                //!< nullspace matrices of the terminal constraint

  DEPRECATED(
      "Do not use this member",
      Scalar dg_;)  //!< Internal data for computing the expected improvement
  DEPRECATED(
      "Do not use this member",
      Scalar dq_;)  //!< Internal data for computing the expected improvement
  DEPRECATED(
      "Do not use this member",
      Scalar dv_;)  //!< Internal data for computing the expected improvement

  using SolverAbstract::acceptstep_;
  using SolverAbstract::alphas_;
  using SolverAbstract::callbacks_;
  using SolverAbstract::cost_;
  using SolverAbstract::cost_try_;
  using SolverAbstract::dfeas_;
  using SolverAbstract::dImpr_;
  using SolverAbstract::dPhi_;
  using SolverAbstract::dPhiexp_;
  using SolverAbstract::dreg_;
  using SolverAbstract::dus_;
  using SolverAbstract::DV_;
  using SolverAbstract::dV_;
  using SolverAbstract::dVexp_;
  using SolverAbstract::dVexp_full_;
  using SolverAbstract::dxs_;
  using SolverAbstract::feas_;
  using SolverAbstract::feasnorm_;
  using SolverAbstract::ffeas_;
  using SolverAbstract::ffeas_try_;
  using SolverAbstract::fs_;
  using SolverAbstract::fs_try_;
  using SolverAbstract::gfeas_;
  using SolverAbstract::gfeas_try_;
  using SolverAbstract::hfeas_;
  using SolverAbstract::hfeas_try_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::iter_;
  using SolverAbstract::merit_;
  using SolverAbstract::nh_T_;
  using SolverAbstract::preg_;
  using SolverAbstract::problem_;
  using SolverAbstract::reg_max_;
  using SolverAbstract::reg_min_;
  using SolverAbstract::steplength_;
  using SolverAbstract::stop_;
  using SolverAbstract::th_acceptstep_;
  using SolverAbstract::th_gaptol_;
  using SolverAbstract::th_stop_;
  using SolverAbstract::us_;
  using SolverAbstract::us_try_;
  using SolverAbstract::xs_;
  using SolverAbstract::xs_try_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/solvers/fddp.hxx"

CROCODDYL_DISABLE_WARNING_DEPRECATED

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SolverFDDPTpl)

CROCODDYL_ENABLE_WARNING_DEPRECATED

#endif  // CROCODDYL_CORE_SOLVERS_FDDP_HPP_
