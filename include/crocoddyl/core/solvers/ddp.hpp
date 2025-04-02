///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_DDP_HPP_

#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Differential Dynamic Programming (DDP) solver
 *
 * The DDP solver computes an optimal trajectory and control commands by
 * iterates running `backwardPass()` and `forwardPass()`. The backward-pass
 * updates locally the quadratic approximation of the problem and computes
 * descent direction. If the warm-start is feasible, then it computes the gaps
 * \f$\mathbf{f}_s\f$ and run a modified Riccati sweep: \f{eqnarray*}
 *   \mathbf{Q}_{\mathbf{x}_k} &=& \mathbf{l}_{\mathbf{x}_k} +
 * \mathbf{f}^\top_{\mathbf{x}_k} (V_{\mathbf{x}_{k+1}} +
 * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
 *   \mathbf{Q}_{\mathbf{u}_k} &=& \mathbf{l}_{\mathbf{u}_k} +
 * \mathbf{f}^\top_{\mathbf{u}_k} (V_{\mathbf{x}_{k+1}} +
 * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
 *   \mathbf{Q}_{\mathbf{xx}_k} &=& \mathbf{l}_{\mathbf{xx}_k} +
 * \mathbf{f}^\top_{\mathbf{x}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{x}_k},\\
 *   \mathbf{Q}_{\mathbf{xu}_k} &=& \mathbf{l}_{\mathbf{xu}_k} +
 * \mathbf{f}^\top_{\mathbf{x}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{u}_k},\\
 *   \mathbf{Q}_{\mathbf{uu}_k} &=& \mathbf{l}_{\mathbf{uu}_k} +
 * \mathbf{f}^\top_{\mathbf{u}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{u}_k}.
 * \f}
 * Then, the forward-pass rollouts this new policy by integrating the system
 * dynamics along a tuple of optimized control commands \f$\mathbf{u}^*_s\f$,
 * i.e. \f{eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
 * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\ \mathbf{\hat{x}}_{k+1} &=&
 * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k). \f}
 *
 * \sa SolverAbstract(), `backwardPass()` and `forwardPass()`
 */
class SolverDDP : public SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef typename MathBaseTpl<double>::MatrixXsRowMajor MatrixXdRowMajor;

  /**
   * @brief Initialize the DDP solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverDDP(std::shared_ptr<ShootingProblem> problem);
  virtual ~SolverDDP();

  virtual bool solve(
      const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
      const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const double init_reg = NAN);
  virtual void computeDirection(const bool recalc = true);
  virtual double tryStep(const double steplength = 1);
  virtual double stoppingCriteria();
  virtual const Eigen::Vector2d& expectedImprovement();
  virtual void resizeData();

  /**
   * @brief Update the Jacobian, Hessian and feasibility of the optimal control
   * problem
   *
   * These derivatives are computed around the guess state and control
   * trajectory. These trajectory can be set by using `setCandidate()`.
   *
   * @return the total cost around the guess trajectory
   */
  virtual double calcDiff();

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
   * @brief Run the forward pass or rollout
   *
   * It rollouts the action model given the computed policy (feedforward terns
   * and feedback gains) by the `backwardPass()`: \f{eqnarray}
   *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0,\\
   *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
   * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\ \mathbf{\hat{x}}_{k+1} &=&
   * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k). \f} We can define
   * different step lengths \f$\alpha\f$.
   *
   * @param stepLength  applied step length (\f$0\leq\alpha\leq1\f$)
   */
  virtual void forwardPass(const double stepLength);

  /**
   * @brief Compute the linear-quadratic approximation of the control
   * Hamiltonian function
   *
   * @param[in] t      Time instance
   * @param[in] model  Action model in the given time instance
   * @param[in] data   Action data in the given time instance
   */
  virtual void computeActionValueFunction(
      const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model,
      const std::shared_ptr<ActionDataAbstract>& data);

  /**
   * @brief Compute the linear-quadratic approximation of the Value function
   *
   * This function is called in the backward pass after updating the local
   * Hamiltonian.
   *
   * @param[in] t      Time instance
   * @param[in] model  Action model in the given time instance
   */
  virtual void computeValueFunction(
      const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model);

  /**
   * @brief Compute the feedforward and feedback terms using a Cholesky
   * decomposition
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
  virtual void computeGains(const std::size_t t);

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
   * @brief Allocate all the internal data needed for the solver
   */
  virtual void allocateData();

  /**
   * @brief Return the regularization factor used to increase the damping value
   */
  double get_reg_incfactor() const;

  /**
   * @brief Return the regularization factor used to decrease the damping value
   */
  double get_reg_decfactor() const;

  /**
   * @brief Return the regularization factor used to decrease / increase it
   */
  DEPRECATED("Use get_reg_incfactor() or get_reg_decfactor()",
             double get_regfactor() const;)

  /**
   * @brief Return the minimum regularization value
   */
  double get_reg_min() const;
  DEPRECATED("Use get_reg_min()", double get_regmin() const);

  /**
   * @brief Return the maximum regularization value
   */
  double get_reg_max() const;
  DEPRECATED("Use get_reg_max()", double get_regmax() const);

  /**
   * @brief Return the set of step lengths using by the line-search procedure
   */
  const std::vector<double>& get_alphas() const;

  /**
   * @brief Return the step-length threshold used to decrease regularization
   */
  double get_th_stepdec() const;

  /**
   * @brief Return the step-length threshold used to increase regularization
   */
  double get_th_stepinc() const;

  /**
   * @brief Return the tolerance of the expected gradient used for testing the
   * step
   */
  double get_th_grad() const;

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
   * @brief Modify the regularization factor used to increase the damping value
   */
  void set_reg_incfactor(const double reg_factor);

  /**
   * @brief Modify the regularization factor used to decrease the damping value
   */
  void set_reg_decfactor(const double reg_factor);

  /**
   * @brief Modify the regularization factor used to decrease / increase it
   */
  DEPRECATED("Use set_reg_incfactor() or set_reg_decfactor()",
             void set_regfactor(const double reg_factor);)

  /**
   * @brief Modify the minimum regularization value
   */
  void set_reg_min(const double regmin);
  DEPRECATED("Use set_reg_min()", void set_regmin(const double regmin));

  /**
   * @brief Modify the maximum regularization value
   */
  void set_reg_max(const double regmax);
  DEPRECATED("Use set_reg_max()", void set_regmax(const double regmax));

  /**
   * @brief Modify the set of step lengths using by the line-search procedure
   */
  void set_alphas(const std::vector<double>& alphas);

  /**
   * @brief Modify the step-length threshold used to decrease regularization
   */
  void set_th_stepdec(const double th_step);

  /**
   * @brief Modify the step-length threshold used to increase regularization
   */
  void set_th_stepinc(const double th_step);

  /**
   * @brief Modify the tolerance of the expected gradient used for testing the
   * step
   */
  void set_th_grad(const double th_grad);

 protected:
  double reg_incfactor_;  //!< Regularization factor used to increase the
                          //!< damping value
  double reg_decfactor_;  //!< Regularization factor used to decrease the
                          //!< damping value
  double reg_min_;        //!< Minimum allowed regularization value
  double reg_max_;        //!< Maximum allowed regularization value

  double cost_try_;  //!< Total cost computed by line-search procedure
  std::vector<Eigen::VectorXd>
      xs_try_;  //!< State trajectory computed by line-search procedure
  std::vector<Eigen::VectorXd>
      us_try_;  //!< Control trajectory computed by line-search procedure
  std::vector<Eigen::VectorXd>
      dx_;  //!< State error during the roll-out/forward-pass (size T)

  // allocate data
  std::vector<Eigen::MatrixXd>
      Vxx_;  //!< Hessian of the Value function \f$\mathbf{V_{xx}}\f$
  Eigen::MatrixXd
      Vxx_tmp_;  //!< Temporary variable for ensuring symmetry of Vxx
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

  Eigen::VectorXd xnext_;      //!< Next state \f$\mathbf{x}^{'}\f$
  MatrixXdRowMajor FxTVxx_p_;  //!< Store the value of
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
  std::vector<double>
      alphas_;      //!< Set of step lengths using by the line-search procedure
  double th_grad_;  //!< Tolerance of the expected gradient used for testing the
                    //!< step
  double
      th_stepdec_;  //!< Step-length threshold used to decrease regularization
  double
      th_stepinc_;  //!< Step-length threshold used to increase regularization
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_HPP_
