///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
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

/**
 * @brief Feasibility-driven Differential Dynamic Programming (FDDP) solver
 *
 * The FDDP solver computes an optimal trajectory and control commands by
 * iterates running `backwardPass()` and `forwardPass()`. The backward pass
 * accepts infeasible guess as described in the `SolverDDP::backwardPass()`.
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
 * \sa `SolverAbstract()`, `backwardPass()`, `forwardPass()`,
 * `expectedImprovement()` and `updateExpectedImprovement()`
 */
class SolverFDDP : public SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef typename MathBaseTpl<double>::MatrixXsRowMajor MatrixXdRowMajor;

  /**
   * @brief Initialize the FDDP solver
   *
   * @param[in] problem     Shooting problem
   * @param[in] dyn_solver  Type of dynamic solver
   */
  explicit SolverFDDP(std::shared_ptr<ShootingProblem> problem,
                      const DynamicsSolverType dyn_solver = FeasShoot);
  virtual ~SolverFDDP();

  /**
   * @copybrief SolverAbstract::solve
   */
  virtual bool solve(
      const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
      const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const double init_reg = NAN);

  /**
   * @copybrief SolverAbstract::computeDirection
   */
  virtual void computeDirection(const bool recalc = true);

  /**
   * @copybrief SolverAbstract::tryStep
   */
  virtual double tryStep(const double steplength = 1, const bool recalc = true);

  /**
   * @copybrief SolverAbstract::stoppingCriteria
   */
  virtual double stoppingCriteria();

  /**
   * @copybrief SolverAbstract::expectedImprovement
   *
   * This function requires to first run `updateExpectedImprovement()`. The
   * expected improvement computation considers the gaps in the dynamics:
   * \f{equation} \Delta J(\alpha) = \Delta_1\alpha +
   * \frac{1}{2}\Delta_2\alpha^2, \f} with \f{eqnarray} \Delta_1 =
   * \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k}
   * +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k}
   * - V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
   *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k +
   * \mathbf{\bar{f}}_k^\top(2 V_{\mathbf{xx}_k}\mathbf{x}_k
   * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). \f}
   */
  virtual const Eigen::Vector2d& expectedImprovement();

  /**
   * @copybrief SolverAbstract::resizeData
   */
  virtual void resizeData();

  /**
   * @brief Update the Jacobian, Hessian and feasibility of the optimal control
   * problem
   *
   * These derivatives are computed around the guess state and control
   * trajectory. These trajectory can be set by using `setCandidate()`.
   */
  virtual void calcDir();

  /**
   * @brief Run the backward pass (Riccati sweep)
   *
   * It assumes that the Jacobian and Hessians of the optimal control problem
   * have been compute (i.e., `calcDiff()`). The backward pass handles
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
  virtual void feasShootForwardPass(const double steplength);

  /**
   * @brief Run the multiple-shooting rollout
   *
   * It rollouts the action model given the multiple-shooting approach described
   * in (TODO: add paper title)
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   * @param[in] recalc  recompute the linear rollout
   */
  virtual void multiShootForwardPass(const double steplength,
                                     const bool recalc);

  /**
   * @brief Run the multiple-shooting rollout with intervals of
   * feasibility-driven search
   *
   * It rollouts the action model given the hybrid-shooting approach described
   * in (TODO: add paper title)
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   * @param[in] recalc  recompute the linear rollout
   */
  virtual void hybridShootForwardPass(const double steplength,
                                      const bool recalc);

  /**
   * @brief Run the classical nonlinear rollout
   *
   * It rollouts the action model given the classical approach in DDP. You can
   * find details in "A second-order gradient method for determining optimal
   * trajectories of non-linear discrete-time systems"
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   */
  virtual void singleShootForwardPass(const double steplength);

  /**
   * @brief Increase the state and control regularization values by a
   * `regfactor_` factor
   */
  void increaseRegularization();

  /**
   * @brief Decrease the state and control regularization values by a
   * `regfactor_` factor
   */
  void decreaseRegularization();

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
   * @brief Return the type of solver used for handling the dynamics constraints
   */
  DynamicsSolverType get_dynamics_solver() const;

  /**
   * @brief Return the set of step lengths using by the line-search procedure
   */
  const std::vector<double>& get_alphas() const;

  /**
   * @brief Return the regularization factor used to increase the damping value
   */
  double get_reg_incfactor() const;

  /**
   * @brief Return the regularization factor used to decrease the damping value
   */
  double get_reg_decfactor() const;

  /**
   * @brief Return the minimum regularization value
   */
  double get_reg_min() const;

  /**
   * @brief Return the maximum regularization value
   */
  double get_reg_max() const;

  /**
   * @brief Return the tolerance of the expected gradient used for testing the
   * step
   */
  double get_th_grad() const;

  /**
   * @brief Return the step-length threshold used to decrease regularization
   */
  double get_th_stepdec() const;

  /**
   * @brief Return the step-length threshold used to increase regularization
   */
  double get_th_stepinc() const;

  /**
   * @brief Return the minimum improvement threshold used to increase
   * regularization
   */
  double get_th_minimprove() const;

  /**
   * @brief Return the threshold used for accepting step along ascent direction
   */
  double get_th_acceptnegstep() const;

  /**
   * @brief Return the threshold used for accepting minimum steps
   */
  double get_th_acceptminstep() const;

  /**
   * @brief Return the rho parameter used in the merit function
   */
  double get_rho() const;

  /**
   * @brief Return the threshold for switching to feasibility
   */
  double get_th_minfeas() const;

  /**
   * @brief Return the estimated penalty parameter that balances relative
   * contribution of the cost function and equality constraints
   */
  double get_upsilon() const;

  /**
   * @brief Return the upsilon decresing factor used to estimate to balance
   * optimality and feasibility
   */
  double get_upsilon_decfactor() const;

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
  const std::vector<Eigen::MatrixXd>& get_Vxx() const;

  /**
   * @brief Return the Hessian of the Value function \f$V_{\mathbf{x}_s}\f$
   */
  const std::vector<Eigen::VectorXd>& get_Vx() const;

  /**
   * @brief Return the Hessian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{xx}_s}\f$
   */
  const std::vector<Eigen::MatrixXd>& get_Qxx() const;

  /**
   * @brief Return the Hessian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{xu}_s}\f$
   */
  const std::vector<Eigen::MatrixXd>& get_Qxu() const;

  /**
   * @brief Return the Hessian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{uu}_s}\f$
   */
  const std::vector<Eigen::MatrixXd>& get_Quu() const;

  /**
   * @brief Return the Jacobian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{x}_s}\f$
   */
  const std::vector<Eigen::VectorXd>& get_Qx() const;

  /**
   * @brief Return the Jacobian of the Hamiltonian function
   * \f$\mathbf{Q}_{\mathbf{u}_s}\f$
   */
  const std::vector<Eigen::VectorXd>& get_Qu() const;

  /**
   * @brief Return the feedback gains \f$\mathbf{K}_{s}\f$
   */
  const std::vector<MatrixXdRowMajor>& get_K() const;

  /**
   * @brief Return the feedforward gains \f$\mathbf{k}_{s}\f$
   */
  const std::vector<Eigen::VectorXd>& get_k() const;

  /**
   * @brief Return the linear update in \f$\delta\mathbf{x}_s\f$
   */
  const std::vector<Eigen::VectorXd>& get_dxs() const;

  /**
   * @brief Return the feedforward gains \f$\delta\mathbf{u}_s\f$
   */
  const std::vector<Eigen::VectorXd>& get_dus() const;

  /**
   * @brief Modify the set of step lengths using by the line-search procedure
   */
  void set_alphas(const std::vector<double>& alphas);

  /**
   * @brief Modify the regularization factor used to increase the damping value
   */
  void set_reg_incfactor(const double reg_factor);

  /**
   * @brief Modify the regularization factor used to decrease the damping value
   */
  void set_reg_decfactor(const double reg_factor);

  /**
   * @brief Modify the minimum regularization value
   */
  void set_reg_min(const double regmin);

  /**
   * @brief Modify the maximum regularization value
   */
  void set_reg_max(const double regmax);

  /**
   * @brief Modify the tolerance of the expected gradient used for testing the
   * step
   */
  void set_th_grad(const double th_grad);

  /**
   * @brief Modify the step-length threshold used to decrease regularization
   */
  void set_th_stepdec(const double th_step);

  /**
   * @brief Modify the step-length threshold used to increase regularization
   */
  void set_th_stepinc(const double th_step);

  /**
   * @brief Modify the minimum improvement threshold used to increase
   * regularization
   */
  void set_th_minimprove(const double th_step);

  /**
   * @brief Modify the threshold used for accepting step along ascent direction
   */
  void set_th_acceptnegstep(const double th_acceptnegstep);

  /**
   * @brief Modify the threshold used for accepting minimum steps
   */
  void set_th_acceptminstep(const double th_acceptminstep);

  /**
   * @brief Modify the rho parameter used in the merit function
   */
  void set_rho(const double rho);

  /**
   * @brief Modify the threshold for switching to feasibility
   */
  void set_th_minfeas(const double th_minfeas);

  /**
   * @brief Modify the upsilon decresing factor used to estimate to balance
   * optimality and feasibility
   */
  void set_upsilon_decfactor(const double th_step);

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

  DynamicsSolverType dyn_solver_;  //!< Type of dynamics solver
  std::vector<double>
      alphas_;  //!< Set of step lengths using by the line-search procedure
  double reg_incfactor_;  //!< Regularization factor used to increase the
                          //!< damping value
  double reg_decfactor_;  //!< Regularization factor used to decrease the
                          //!< damping value
  double reg_min_;        //!< Minimum allowed regularization value
  double reg_max_;        //!< Maximum allowed regularization value
  double th_grad_;  //!< Tolerance of the expected gradient used for testing the
                    //!< step
  double
      th_stepdec_;  //!< Step-length threshold used to decrease regularization
  double
      th_stepinc_;  //!< Step-length threshold used to increase regularization
  double th_minimprove_;     //!< Minimum improvement threshold used in the
                             //!< regularization scheme
  double th_acceptnegstep_;  //!< Threshold used for accepting step along ascent
                             //!< direction
  double th_acceptminstep_;  //!< Threshold used for accepting step along with a
                             //!< minimum length
  double rho_;         //!< Parameter used in the merit function to predict the
                       //!< expected reduction
  double th_minfeas_;  //!< Threshold for switching to feasibility
  double
      upsilon_;  //!< Estimated penalty parameter that balances relative
                 //!< contribution of the cost function and equality constraints
  double upsilon_decfactor_;  //!< Estimated penalty parameter factor used to
                              //!< decrease its value
  bool zero_upsilon_;  //!< True if we wish to set estimated penalty parameter
                       //!< (upsilon) to zero when solve is called.
  std::vector<std::size_t> Ts_;  //!< Index that describes the hybrid shoots

  // allocate data
  double dImpr_;  //!< Reduction in the iteration improvement (i.e., maximum
                  //!< between cost and merit values)
  Eigen::MatrixXd
      Vxx_tmp_;  //!< Temporary variable for ensuring symmetry of Vxx
  std::vector<Eigen::MatrixXd>
      Vxx_;  //!< Hessian of the Value function \f$\mathbf{V_{xx}}\f$
  std::vector<Eigen::VectorXd>
      Vxx_f_;  //!< Hessian of the Value function times the gap
               //!< \f$\mathbf{V_{xx} \bar{f}}\f$
  std::vector<Eigen::VectorXd>
      Vx_;  //!< Gradient of the Value function \f$\mathbf{V_x}\f$
  std::vector<Eigen::MatrixXd>
      Qxx_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{xx}}\f$
  std::vector<Eigen::MatrixXd>
      Qxu_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{xu}}\f$
  std::vector<Eigen::MatrixXd>
      Quu_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{uu}}\f$
  std::vector<Eigen::VectorXd>
      Qx_;  //!< Gradient of the Hamiltonian \f$\mathbf{Q_x}\f$
  std::vector<Eigen::VectorXd>
      Qu_;  //!< Gradient of the Hamiltonian \f$\mathbf{Q_u}\f$
  std::vector<MatrixXdRowMajor> K_;  //!< Feedback gains \f$\mathbf{K}\f$
  std::vector<Eigen::VectorXd> k_;   //!< Feed-forward terms \f$\mathbf{l}\f$
  std::vector<Eigen::VectorXd>
      dx_;  //!< State error during the roll-out/forward-pass (size T)
  std::vector<Eigen::VectorXd> dxs_;  //!< Linear state direction (size T + 1)
  std::vector<Eigen::VectorXd> dus_;  //!< Linear control direction (size T)
  std::vector<MatrixXdRowMajor>
      FxTVxx_p_;  //!< Store the value of
                  //!< \f$\mathbf{f_x}^T\mathbf{V_{xx}}^{'}\f$
  std::vector<MatrixXdRowMajor>
      FuTVxx_p_;             //!< Store the values of
                             //!< \f$\mathbf{f_u}^T\mathbf{V_{xx}}^{'}\f$
                             //!< per each running node
  Eigen::VectorXd fTVxx_p_;  //!< Store the value of
                             //!< \f$\mathbf{\bar{f}}^T\mathbf{V_{xx}}^{'}\f$
  std::vector<Eigen::LLT<Eigen::MatrixXd> > Quu_llt_;  //!< Cholesky LLT solver
  std::vector<Eigen::VectorXd>
      Quuk_;  //!< Store the values of \f$\mathbf{Q_{uu}\mathbf{k}} per each
              //!< running node

  DEPRECATED(
      "Do not use this member",
      double dg_;)  //!< Internal data for computing the expected improvement
  DEPRECATED(
      "Do not use this member",
      double dq_;)  //!< Internal data for computing the expected improvement
  DEPRECATED(
      "Do not use this member",
      double dv_;)  //!< Internal data for computing the expected improvement

 private:
  double computeFeasibility(const std::vector<Eigen::VectorXd>& fs);
  bool acceptstep_;
  bool recalcdir_;
  bool recalcstep_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_FDDP_HPP_
