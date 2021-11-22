///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVER_BASE_HPP_
#define CROCODDYL_CORE_SOLVER_BASE_HPP_

#include <vector>

#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

class CallbackAbstract;  // forward declaration
static std::vector<Eigen::VectorXd> DEFAULT_VECTOR;

/**
 * @brief Abstract class for optimal control solvers
 *
 * A solver resolves an optimal control solver of the form
 * \f{eqnarray*}{
 * \begin{Bmatrix}
 * 	\mathbf{x}^*_0,\cdots,\mathbf{x}^*_{T} \\
 * 	\mathbf{u}^*_0,\cdots,\mathbf{u}^*_{T-1}
 * \end{Bmatrix} =
 * \arg\min_{\mathbf{x}_s,\mathbf{u}_s} && l_T (\mathbf{x}_T) + \sum_{k=0}^{T-1} l_k(\mathbf{x}_t,\mathbf{u}_t) \\
 * \operatorname{subject}\,\operatorname{to} && \mathbf{x}_0 = \mathbf{\tilde{x}}_0\\
 * &&  \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)\\
 * &&  \mathbf{x}_k\in\mathcal{X}, \mathbf{u}_k\in\mathcal{U}
 * \f}
 * where \f$l_T(\mathbf{x}_T)\f$, \f$l_k(\mathbf{x}_t,\mathbf{u}_t)\f$ are the terminal and running cost functions,
 * respectively, \f$\mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)\f$ describes evolution of the system, and state and
 * control admissible sets are defined by \f$\mathbf{x}_k\in\mathcal{X}\f$, \f$\mathbf{u}_k\in\mathcal{U}\f$.
 * An action model, defined in the shooting problem, describes each node \f$k\f$. Inside the action model, we
 * specialize the cost functions, the system evolution and the admissible sets.
 *
 * The main routines are `computeDirection()` and `tryStep()`. The former finds a search direction and typically
 * computes the derivatives of each action model. The latter rollout the dynamics and cost (i.e., the action)
 * to try the search direction found by `computeDirection`. Both functions used the current guess defined by
 * `setCandidate()`. Finally, `solve()` function is used to define when the search direction and length are
 * computed in each iterate. It also describes the globalization strategy (i.e., regularization) of the
 * numerical optimization.
 *
 * \sa `solve()`, `computeDirection()`, `tryStep()`, `stoppingCriteria()`
 */
class SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverAbstract(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverAbstract();

  /**
   * @brief Compute the optimal trajectory \f$\mathbf{x}^*_s,\mathbf{u}^*_s\f$ as lists of \f$T+1\f$ and \f$T\f$ terms
   *
   * From an initial guess \p init_xs, \p init_us (feasible or not), iterate over `computeDirection()` and `tryStep()`
   * until `stoppingCriteria()` is below threshold. It also describes the globalization strategy used during the
   * numerical optimization.
   *
   * @param[in] init_xs     initial guess for state trajectory with \f$T+1\f$ elements (default [])
   * @param[in] init_us     initial guess for control trajectory with \f$T\f$ elements (default [])
   * @param[in] maxiter     maximum allowed number of iterations (default 100)
   * @param[in] isFeasible  true if the \p init_xs are obtained from integrating the \p init_us (rollout) (default
   * false)
   * @param[in] regInit     initial guess for the regularization value. Very low values are typical used with very
   * good guess points (init_xs, init_us)
   * @return A boolean that describes if convergence was reached.
   */
  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
                     const bool is_feasible = false, const double reg_init = 1e-9) = 0;

  /**
   * @brief Compute the search direction \f$(\delta\mathbf{x}^k,\delta\mathbf{u}^k)\f$ for the current guess
   * \f$(\mathbf{x}^k_s,\mathbf{u}^k_s)\f$.
   *
   * You must call `setCandidate()` first in order to define the current guess. A current guess defines a state
   * and control trajectory \f$(\mathbf{x}^k_s,\mathbf{u}^k_s)\f$ of \f$T+1\f$ and \f$T\f$ elements, respectively.
   *
   * @param[in] recalc  true for recalculating the derivatives at current state and control
   * @return  The search direction \f$(\delta\mathbf{x},\delta\mathbf{u})\f$ and the dual lambdas as lists of
   * \f$T+1\f$, \f$T\f$ and \f$T+1\f$ lengths, respectively
   */
  virtual void computeDirection(const bool recalc) = 0;

  /**
   * @brief Try a predefined step length \f$\alpha\f$ and compute its cost improvement \f$dV\f$.
   *
   * It uses the search direction found by `computeDirection()` to try a determined step length \f$\alpha\f$.
   * Therefore, it assumes that we have run `computeDirection()` first. Additionally, it returns the cost improvement
   * \f$dV\f$ along the predefined step length \f$\alpha\f$.
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$)
   * @return  the cost improvement
   */
  virtual double tryStep(const double steplength = 1) = 0;

  /**
   * @brief Return a positive value that quantifies the algorithm termination
   *
   * These values typically represents the gradient norm which tell us that it's been reached the local minima.
   * The stopping criteria strictly speaking depends on  the search direction (calculated by `computeDirection()`) but
   * it could also depend on the chosen step length, tested by `tryStep()`.
   */
  virtual double stoppingCriteria() = 0;

  /**
   * @brief Return the expected improvement \f$dV_{exp}\f$ from a given current search direction
   * \f$(\delta\mathbf{x}^k,\delta\mathbf{u}^k)\f$
   *
   * For computing the expected improvement, you need to compute the search direction first via `computeDirection()`.
   */
  virtual const Eigen::Vector2d& expectedImprovement() = 0;

  /**
   * @brief Resizing the solver data
   *
   * If the shooting problem has changed after construction, then this function resizes all the data before starting
   * resolve the problem.
   */
  virtual void resizeData();

  /**
   * @brief Compute the dynamic feasibility \f$\|\mathbf{f}_{\mathbf{s}}\|_{\infty,1}\f$ for
   * the current guess \f$(\mathbf{x}^k,\mathbf{u}^k)\f$
   *
   * The feasibility can be computed using the computed using the \f$\ell_\infty\f$ and \f$\ell_1\f$ norms.
   * By default we use the \f$\ell_\infty\f$ norm; however, we can use the \f$\ell_1\f$ norm via `set_inffeas()`.
   * Note that \f$\mathbf{f}_{\mathbf{s}}\f$ are the gaps on the dynamics, which are computed at each node as
   * \f$\mathbf{x}^{'}-\mathbf{f}(\mathbf{x},\mathbf{u})\f$.
   */
  double computeDynamicFeasibility();

  /**
   * @brief Set the solver candidate trajectories \f$(\mathbf{x}_s,\mathbf{u}_s)\f$
   *
   * The solver candidates are defined as a state and control trajectories \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ of
   * \f$T+1\f$ and \f$T\f$ elements, respectively. Additionally, we need to define the dynamic feasibility of the
   * \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ pair. Note that the trajectories are feasible if \f$\mathbf{x}_s\f$ is the
   * resulting trajectory from the system rollout with \f$\mathbf{u}_s\f$ inputs.
   *
   * @param[in] xs          state trajectory of \f$T+1\f$ elements (default [])
   * @param[in] us          control trajectory of \f$T\f$ elements (default [])
   * @param[in] isFeasible  true if the \p xs are obtained from integrating the \p us (rollout)
   */
  void setCandidate(const std::vector<Eigen::VectorXd>& xs_warm = DEFAULT_VECTOR,
                    const std::vector<Eigen::VectorXd>& us_warm = DEFAULT_VECTOR, const bool is_feasible = false);

  /**
   * @brief Set a list of callback functions using for the solver diagnostic
   *
   * Each iteration, the solver calls these set of functions in order to allowed user the diagnostic of its
   * performance.
   *
   * @param  callbacks  set of callback functions
   */
  void setCallbacks(const std::vector<boost::shared_ptr<CallbackAbstract> >& callbacks);

  /**
   * @brief Return the list of callback functions using for diagnostic
   */
  const std::vector<boost::shared_ptr<CallbackAbstract> >& getCallbacks() const;

  /**
   * @brief Return the shooting problem
   */
  const boost::shared_ptr<ShootingProblem>& get_problem() const;

  /**
   * @brief Return the state trajectory \f$\mathbf{x}_s\f$
   */
  const std::vector<Eigen::VectorXd>& get_xs() const;

  /**
   * @brief Return the control trajectory \f$\mathbf{u}_s\f$
   */
  const std::vector<Eigen::VectorXd>& get_us() const;

  /**
   * @brief Return the gaps \f$\mathbf{f}_{s}\f$
   */
  const std::vector<Eigen::VectorXd>& get_fs() const;

  /**
   * @brief Return the feasibility status of the \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ trajectory
   */
  bool get_is_feasible() const;

  /**
   * @brief Return the total cost
   */
  double get_cost() const;

  /**
   * @brief Return the value computed by `stoppingCriteria()`
   */
  double get_stop() const;

  /**
   * @brief Return the LQ approximation of the expected improvement
   */
  const Eigen::Vector2d& get_d() const;

  /**
   * @brief Return the state regularization value
   */
  double get_xreg() const;

  /**
   * @brief Return the control regularization value
   */
  double get_ureg() const;

  /**
   * @brief Return the step length \f$\alpha\f$
   */
  double get_steplength() const;

  /**
   * @brief Return the cost reduction \f$dV\f$
   */
  double get_dV() const;

  /**
   * @brief Return the expected cost reduction \f$dV_{exp}\f$
   */
  double get_dVexp() const;

  /**
   * @brief Return the threshold used for accepting a step
   */
  double get_th_acceptstep() const;

  /**
   * @brief Return the tolerance for stopping the algorithm
   */
  double get_th_stop() const;

  /**
   * @brief Return the number of iterations performed by the solver
   */
  std::size_t get_iter() const;

  /**
   * @brief Return the threshold for accepting a gap as non-zero
   */
  double get_th_gaptol() const;

  /**
   * @brief Return the feasibility of the dynamic constraints \f$\|\mathbf{f}_{\mathbf{s}}\|_{\infty,1}\f$ of the
   * current guess
   */
  double get_ffeas() const;

  /**
   * @brief Return the norm used for the computing the feasibility (true for \f$\ell_\infty\f$, false for \f$\ell_1\f$)
   */
  bool get_inffeas() const;

  /**
   * @brief Modify the state trajectory \f$\mathbf{x}_s\f$
   */
  void set_xs(const std::vector<Eigen::VectorXd>& xs);

  /**
   * @brief Modify the control trajectory \f$\mathbf{u}_s\f$
   */
  void set_us(const std::vector<Eigen::VectorXd>& us);

  /**
   * @brief Modify the state regularization value
   */
  void set_xreg(const double xreg);

  /**
   * @brief Modify the control regularization value
   */
  void set_ureg(const double ureg);

  /**
   * @brief Modify the threshold used for accepting step
   */
  void set_th_acceptstep(const double th_acceptstep);

  /**
   * @brief Modify the tolerance for stopping the algorithm
   */
  void set_th_stop(const double th_stop);

  /**
   * @brief Modify the threshold for accepting a gap as non-zero
   */
  void set_th_gaptol(const double th_gaptol);

  /**
   * @brief Modify the current norm used for computed the feasibility
   */
  void set_inffeas(const bool inffeas);

 protected:
  boost::shared_ptr<ShootingProblem> problem_;                   //!< optimal control problem
  std::vector<Eigen::VectorXd> xs_;                              //!< State trajectory
  std::vector<Eigen::VectorXd> us_;                              //!< Control trajectory
  std::vector<Eigen::VectorXd> fs_;                              //!< Gaps/defects between shooting nodes
  std::vector<boost::shared_ptr<CallbackAbstract> > callbacks_;  //!< Callback functions
  bool is_feasible_;                                             //!< Label that indicates is the iteration is feasible
  bool was_feasible_;     //!< Label that indicates in the previous iterate was feasible
  double cost_;           //!< Total cost
  double stop_;           //!< Value computed by `stoppingCriteria()`
  Eigen::Vector2d d_;     //!< LQ approximation of the expected improvement
  double xreg_;           //!< Current state regularization value
  double ureg_;           //!< Current control regularization values
  double steplength_;     //!< Current applied step-length
  double dV_;             //!< Cost reduction obtained by `tryStep()`
  double dVexp_;          //!< Expected cost reduction
  double th_acceptstep_;  //!< Threshold used for accepting step
  double th_stop_;        //!< Tolerance for stopping the algorithm
  std::size_t iter_;      //!< Number of iteration performed by the solver
  double th_gaptol_;      //!< Threshold limit to check non-zero gaps
  double ffeas_;          //!< Feasibility of the dynamic constraints
  bool inffeas_;     //!< True indicates if we use l-inf norm for computing the feasibility, otherwise false represents
                     //!< the l-1 norm
  double tmp_feas_;  //!< Temporal variables used for computed the feasibility
};

/**
 * @brief Abstract class for solver callbacks
 *
 * A callback is used to diagnostic the behaviour of our solver in each iteration of it. For instance, it can be used
 * to print values, record data or display motions.
 */
class CallbackAbstract {
 public:
  /**
   * @brief Initialize the callback function
   */
  CallbackAbstract() {}
  virtual ~CallbackAbstract() {}

  /**
   * @brief Run the callback function given a solver
   *
   * @param[in]  solver solver to be diagnostic
   */
  virtual void operator()(SolverAbstract& solver) = 0;
};

bool raiseIfNaN(const double value);

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVER_BASE_HPP_
