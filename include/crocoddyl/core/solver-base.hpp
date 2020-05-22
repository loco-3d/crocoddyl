///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVER_BASE_HPP_
#define CROCODDYL_CORE_SOLVER_BASE_HPP_

#include <vector>

#include "crocoddyl/core/optctrl/shooting.hpp"

namespace crocoddyl {

class CallbackAbstract;  // forward declaration
static std::vector<Eigen::VectorXd> DEFAULT_VECTOR;

// \begin{eqnarray*}
// \mathbf{X}^*(\mathbf{\tilde{x}}_0), \mathbf{U}^*(\mathbf{\tilde{x}}_0) =
// \begin{Bmatrix}
// 	\mathbf{x}^*_0,\cdots,\mathbf{x}^*_N \\
// 	\mathbf{u}^*_0,\cdots,\mathbf{u}^*_{N-1}
// \end{Bmatrix} =
// \argmin_{\mathbf{X},\mathbf{U}} && l_N (\mathbf{x}_N) +
// 	\sum_{k=0}^{N-1} l_k(\mathbf{x}_t,\mathbf{u}_t) \\
// \st && \mathbf{x}_0 = \mathbf{\tilde{x}}_0\\
//     &&  \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)
// \end{eqnarray*}

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
 * computes the derivatives of each action model. The latter rollout the dynamics and cost (i.e. the action)
 * to try the search direction found by computeDirection. Both functions used the current guess defined by
 * setCandidate. Finally solve function is used to define when the search direction and length are
 * computed in each iterate. It also describes the globalization strategy (i.e. regularization) of the
 * numerical optimization.
 *
 * \sa `computeDirection()` and `tryStep()`
 */
class SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the solver
   *
   * @param[in] problem  Shooting problem
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
   * @param[in]  init_xs     initial guess for state trajectory with \f$T+1\f$ elements (default [])
   * @param[in]  init_us     initial guess for control trajectory with \f$T\f$ elements (default [])
   * @param[in]  maxiter     maximun allowed number of iterations (default 100)
   * @param[in]  isFeasible  true if the \p init_xs are obtained from integrating the \p init_us (rollout) (default
   * false)
   * @param[in]  regInit     initial guess for the regularization value. Very low values are typical used with very
   * good guess points (init_xs, init_us)
   * @return A boolean that describes if convergence was reached.")
   */
  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t& maxiter = 100,
                     const bool& is_feasible = false, const double& reg_init = 1e-9) = 0;

  /**
   * @brief Compute the search direction \f$(\delta\mathbf{x},\delta\mathbf{u})\f$ for the current guess
   * \f$(\mathbf{x}_s,\mathbf{u}_s)\f$
   *
   * You must call setCandidate first in order to define the current guess. A current guess defines a state
   * and control trajectory \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ of \f$T+1\f$ and \f$T\f$ elements, respectively.
   *
   * @param[in] recalc  True for recalculating the derivatives at current state and control
   * @return  The search direction \f$(\delta\mathbf{x},\delta\mathbf{u})\f$ and the dual lambdas as lists of
   * \f$T+1\f$, \f$T\f$ and \f$T+1\f$ lengths, respectively
   */
  virtual void computeDirection(const bool& recalc) = 0;

  /**
   * @brief Try a predefined step length and compute its cost improvement
   *
   * It uses the search direction found by `computeDirection()` to try a determined step length \f$\alpha\f$; so you
   * need to run first `computeDirection()`. Additionally it returns the cost improvement along the predefined step
   * length.
   *
   * @param[in]  stepLength  step length
   * @return  The cost improvement
   */
  virtual double tryStep(const double& step_length = 1) = 0;

  /**
   * @brief Return a positive value that quantifies the algorithm termination
   *
   * These values typically represents the gradient norm which tell us that it's been reached the local minima.
   * This function is used to evaluate the algorithm convergence. The stopping criteria strictly speaking depends on
   * the search direction (calculated by `computeDirection()`) but it could also depend on the chosen step length,
   * tested by `tryStep()`.
   */
  virtual double stoppingCriteria() = 0;

  /**
   * @brief Return the expected improvement from a given current search direction
   *
   * For computing the expected improvement, you need to compute first the search direction by running
   * `computeDirection()`.
   */
  virtual const Eigen::Vector2d& expectedImprovement() = 0;

  /**
   * @brief Set the solver candidate warm-point values \f$(\mathbf{x}_s,\mathbf{u}_s)\f$
   *
   * The solver candidates are defined as a state and control trajectories \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ of
   * \f$T+1\f$ and \f$T\f$ elements, respectively. Additionally, we need to define is \f$(\mathbf{x}_s,\mathbf{u}_s)\f$
   * pair is feasible, this means that the dynamics rollout give us produces \f$\mathbf{x}_s\f$.
   *
   * @param[in]  xs          state trajectory of \f$T+1\f$ elements (default [])
   * @param[in]  us          control trajectory of \f$T\f$ elements (default [])
   * @param[in]  isFeasible  true if the \p xs are obtained from integrating the \p us (rollout)
   */
  void setCandidate(const std::vector<Eigen::VectorXd>& xs_warm = DEFAULT_VECTOR,
                    const std::vector<Eigen::VectorXd>& us_warm = DEFAULT_VECTOR, const bool& is_feasible = false);

  /**
   * @brief Set a list of callback functions using for diagnostic
   *
   * Each iteration, the solver calls these set of functions in order to allowed user the diagnostic of its
   * performance.
   *
   * @param  callbacks  set of callback functions
   */
  void setCallbacks(const std::vector<boost::shared_ptr<CallbackAbstract> >& callbacks);

  /**
   * @brief "Return the list of callback functions using for diagnostic
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
   * @brief Return the feasibility status of the \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ trajectory
   */
  const bool& get_is_feasible() const;

  /**
   * @brief Return the total cost
   */
  const double& get_cost() const;

  /**
   * @brief Return the value computed by `stoppingCriteria()`
   */
  const double& get_stop() const;

  /**
   * @brief Return the LQ approximation of the expected improvement
   */
  const Eigen::Vector2d& get_d() const;

  /**
   * @brief Return the state regularization value
   */
  const double& get_xreg() const;

  /**
   * @brief Return the control regularization value
   */
  const double& get_ureg() const;

  /**
   * @brief Return the step length
   */
  const double& get_steplength() const;

  /**
   * @brief Return the cost reduction
   */
  const double& get_dV() const;

  /**
   * @brief Return the expected cost reduction
   */
  const double& get_dVexp() const;

  /**
   * @brief Return the threshold used for accepting a step
   */
  const double& get_th_acceptstep() const;

  /**
   * @brief Return the tolerance for stopping the algorithm
   */
  const double& get_th_stop() const;

  /**
   * @brief Return the number of iterations performed by the solver
   */
  const std::size_t& get_iter() const;

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
  void set_xreg(const double& xreg);

  /**
   * @brief Modify the control regularization value
   */
  void set_ureg(const double& ureg);

  /**
   * @brief Modify the threshold used for accepting step
   */
  void set_th_acceptstep(const double& th_acceptstep);

  /**
   * @brief Modify the tolerance for stopping the algorithm
   */
  void set_th_stop(const double& th_stop);

 protected:
  boost::shared_ptr<ShootingProblem> problem_;                   //!< optimal control problem
  std::vector<Eigen::VectorXd> xs_;                              //!< State trajectory
  std::vector<Eigen::VectorXd> us_;                              //!< Control trajectory
  std::vector<boost::shared_ptr<CallbackAbstract> > callbacks_;  //!< Callback functions
  bool is_feasible_;                                             //!< Label that indicates is the iteration is feasible
  double cost_;                                                  //!< Total cost
  double stop_;                                                  //!< Value computed by `stoppingCriteria()`
  Eigen::Vector2d d_;                                            //!< LQ approximation of the expected improvement
  double xreg_;                                                  //!< Current state regularization value
  double ureg_;                                                  //!< Current control regularization values
  double steplength_;                                            //!< Current applied step-length
  double dV_;                                                    //!< Cost reduction obtained by `tryStep()`
  double dVexp_;                                                 //!< Expected cost reduction
  double th_acceptstep_;                                         //!< Threshold used for accepting step
  double th_stop_;                                               //!< Tolerance for stopping the algorithm
  std::size_t iter_;                                             //!< Number of iteration performed by the solver
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

bool raiseIfNaN(const double& value);

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVER_BASE_HPP_
