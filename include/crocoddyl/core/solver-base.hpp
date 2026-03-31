///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVER_BASE_HPP_
#define CROCODDYL_CORE_SOLVER_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DefaultVector {
  static const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> value;
};

template <typename Scalar>
const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>
    DefaultVector<Scalar>::value = {};

#define DEFAULT_VECTOR DefaultVector<double>::value

enum FeasibilityNorm { LInf = 0, L1 = 1, L2 = 2 };

class SolverBase {
 public:
  virtual ~SolverBase() = default;

  CROCODDYL_BASE_CAST(SolverBase, SolverAbstractTpl)
};

/**
 * @brief Abstract class for optimal control solvers
 *
 * A solver resolves an optimal control solver of the form
 * \f{eqnarray*}{
 * \begin{Bmatrix}
 * 	\mathbf{x}^*_0,\cdots,\mathbf{x}^*_{T} \\
 * 	\mathbf{u}^*_0,\cdots,\mathbf{u}^*_{T-1}
 * \end{Bmatrix} =
 * \arg\min_{\mathbf{x}_s,\mathbf{u}_s} && l_T (\mathbf{x}_T) + \sum_{k=0}^{T-1}
 * l_k(\mathbf{x}_t,\mathbf{u}_t) \\
 * \operatorname{subject}\,\operatorname{to} && \mathbf{x}_0 =
 * \mathbf{\tilde{x}}_0\\
 * &&  \mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)\\
 * &&  \mathbf{x}_k\in\mathcal{X}, \mathbf{u}_k\in\mathcal{U}
 * \f}
 * where \f$l_T(\mathbf{x}_T)\f$, \f$l_k(\mathbf{x}_t,\mathbf{u}_t)\f$ are the
 * terminal and running cost functions, respectively,
 * \f$\mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)\f$ describes evolution of the
 * system, and state and control admissible sets are defined by
 * \f$\mathbf{x}_k\in\mathcal{X}\f$, \f$\mathbf{u}_k\in\mathcal{U}\f$. An action
 * model, defined in the shooting problem, describes each node \f$k\f$. Inside
 * the action model, we specialize the cost functions, the system evolution and
 * the admissible sets.
 *
 * The main routines are `computeDirection()` and `tryStep()`. The former finds
 * a search direction and typically computes the derivatives of each action
 * model. The latter rollout the dynamics and cost (i.e., the action) to try the
 * search direction found by `computeDirection`. Both functions used the current
 * guess defined by `setCandidate()`. Finally, `solve()` function is used to
 * define when the search direction and length are computed in each iterate. It
 * also describes the globalization strategy (i.e., regularization) of the
 * numerical optimization.
 *
 * \sa `solve()`, `computeDirection()`, `tryStep()`, `stoppingCriteria()`
 */
template <typename _Scalar>
class SolverAbstractTpl : public SolverBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ShootingProblem::ActionModelAbstract ActionModelAbstract;
  typedef typename ShootingProblem::ActionDataAbstract ActionDataAbstract;
  typedef CallbackAbstractTpl<Scalar> CallbackAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;

  /**
   * @brief Initialize the solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverAbstractTpl(std::shared_ptr<ShootingProblem> problem);
  virtual ~SolverAbstractTpl() = default;

  /**
   * @brief Compute the optimal trajectory \f$\mathbf{x}^*_s,\mathbf{u}^*_s\f$
   * as lists of \f$T+1\f$ and \f$T\f$ terms
   *
   * From an initial guess \p init_xs, \p init_us (feasible or not), iterate
   * over `computeDirection()` and `tryStep()` until `stoppingCriteria()` is
   * below threshold. It also describes the globalization strategy used during
   * the numerical optimization.
   *
   * @param[in] init_xs      initial guess for state trajectory with \f$T+1\f$
   * elements (default [])
   * @param[in] init_us      initial guess for control trajectory with \f$T\f$
   * elements (default [])
   * @param[in] maxiter      maximum allowed number of iterations (default 100)
   * @param[in] is_feasible  true if the \p init_xs are obtained from
   * integrating the \p init_us (rollout) (default false)
   * @param[in] init_reg     initial guess for the regularization value. Very
   * low values are typical used with very good guess points (default 1e-9).
   * @return A boolean that describes if convergence was reached.
   */
  virtual bool solve(
      const std::vector<VectorXs>& init_xs = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& init_us = DefaultVector<Scalar>::value,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const Scalar reg_init = std::numeric_limits<Scalar>::quiet_NaN());

  /**
   * @brief Compute the search direction
   * \f$(\delta\mathbf{x}^k,\delta\mathbf{u}^k)\f$ for the current guess
   * \f$(\mathbf{x}^k_s,\mathbf{u}^k_s)\f$.
   *
   * You must call `setCandidate()` first in order to define the current guess.
   * A current guess defines a state and control trajectory
   * \f$(\mathbf{x}^k_s,\mathbf{u}^k_s)\f$ of \f$T+1\f$ and \f$T\f$ elements,
   * respectively. When \p recalc is true (the default in the provided
   * algorithms), the method refreshes the linearization by calling
   * `calcDir()` before running the backward pass. Setting \p recalc to false
   * reuses the most recent derivatives, which is useful when `calcDir()` has
   * already been triggered explicitly. The resulting state and control
   * variations, together with the dynamics multipliers, are stored in the
   * solver data structures (e.g., `dxs_`, `dus_`, and `lambdas_`).
   *
   * @param[in] recalc  true to refresh the derivatives before computing the
   * search direction
   */
  virtual void computeDirection(const bool recalc) = 0;

  /**
   * @brief Try a predefined step length \f$\alpha\f$ and compute its cost
   * improvement \f$dV\f$ and merit improvement \f$d\Phi\f$.
   *
   * It uses the search direction found by `computeDirection()` to try a
   * determined step length \f$\alpha\f$. Therefore, it assumes that we have run
   * `computeDirection()` first. Additionally, it returns the cost improvement
   * \f$dV\f$ along the predefined step length \f$\alpha\f$. Internally, it
   * updates the cost improvement \f$dV\f$ and merit improvement \f$d\Phi\f$.
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$, with 1
   * as default)
   * @return  the cost improvement
   */
  virtual Scalar tryStep(const Scalar steplength = Scalar(1.));

  /**
   * @brief Compute new candidate solution using step length.
   *
   * This method updates the candidate primal and dual variables by taking a
   * step from the current solution along the computed search direction, scaled
   * by the step length `alpha`. It prepares a candidate point that is
   * evaluated before being accepted by the solver.
   *
   * @param[in] steplength  applied step length (\f$0\leq\alpha\leq1\f$, with 1
   * as default)
   */
  virtual void computeCandidate(const Scalar steplength = Scalar(1.));

  /**
   * @brief Return a positive value that quantifies the algorithm termination
   *
   * These values typically represents the gradient norm which tell us that it's
   * been reached the local minima. The stopping criteria strictly speaking
   * depends on  the search direction (calculated by `computeDirection()`) but
   * it could also depend on the chosen step length, tested by `tryStep()`.
   */
  virtual Scalar stoppingCriteria() = 0;

  /**
   * @brief Return the expected improvement \f$dV_{exp}\f$ from a given current
   * search direction \f$(\delta\mathbf{x}^k,\delta\mathbf{u}^k)\f$
   *
   * For computing the expected improvement, you need to compute the search
   * direction first via `computeDirection()`. The quadratic improvement model
   * is described as dV = DV_0 + DV_1 + 0.5 * DV_2, where DV_0, DV_1, and DV_2
   * are the constant, linear, and quadratic terms, respectively.
   */
  virtual Vector3s expectedImprovement() = 0;

  /**
   * @brief Compute the merit function improvement for the current step.
   *
   * This function computes the current merit function improvement given the
   * current cost improvement and feasibility improvement. The feasibility
   * improvement are stored in dfeas_.
   */
  virtual void computeMeritFunctionImprovement();

  /**
   * @brief Compute the expected merit function improvement for the current
   * step.
   *
   * This function computes the expected merit function improvement given the
   * expected cost improvement and feasibility improvement. The feasibility
   * improvement are stored in dfeas_.
   */
  virtual void computeExpectedMeritFunctionImprovement();

  /**
   * @brief Update the merit function value for the current guess
   */
  virtual void updateMeritFunction();

  /**
   * @brief Check if we should accept or not the step
   *
   * @return True if we should accept the step. False otherwise
   */
  virtual bool checkAcceptance();

  /**
   * @brief Refresh the linearization of the optimal control problem around the
   * current candidate.
   *
   * This routine evaluates the shooting problem along the candidate
   * trajectories `(xs_, us_)` to update the cost, dynamics, and constraint
   * derivatives together with the feasibility metrics used by the solver. The
   * main `solve()` loop invokes `calcDir()` automatically whenever a new
   * linearization is required, but the method remains available when driving
   * the solver manually (e.g., before calling `computeDirection(false)`).
   */
  virtual void calcDir();

  /**
   * @brief Set the solver candidate trajectories
   * \f$(\mathbf{x}_s,\mathbf{u}_s)\f$
   *
   * The solver candidates are defined as a state and control trajectories
   * \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ of \f$T+1\f$ and \f$T\f$ elements,
   * respectively. Additionally, we need to define the dynamic feasibility of
   * the \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ pair. Note that the trajectories are
   * feasible if \f$\mathbf{x}_s\f$ is the resulting trajectory from the system
   * rollout with \f$\mathbf{u}_s\f$ inputs. Updating the candidate invalidates
   * any previously computed linearization; the next call to
   * `computeDirection()` will therefore refresh the derivatives on demand.
   *
   * @param[in] xs          state trajectory of \f$T+1\f$ elements (default [])
   * @param[in] us          control trajectory of \f$T\f$ elements (default [])
   * @param[in] isFeasible  true if the \p xs are obtained from integrating the
   * \p us (rollout)
   */
  virtual void setCandidate(
      const std::vector<VectorXs>& xs_warm = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& us_warm = DefaultVector<Scalar>::value,
      const bool is_feasible = false);

  /**
   * @brief Update the candidate solution: cost, feasibilities, and merit value
   */
  virtual void updateCandidate();

  /**
   * @brief Criteria used to decrease regularization
   */
  virtual bool decreaseRegularizationCriteria();

  /**
   * @brief Criteria used to increase regularization
   */
  virtual bool increaseRegularizationCriteria();

  /**
   * @brief Increase the state and control regularization values by a
   * `regfactor_` factor
   */
  virtual void increaseRegularization();

  /**
   * @brief Decrease the state and control regularization values by a
   * `regfactor_` factor
   */
  virtual void decreaseRegularization();

  /**
   * @brief Resizing the solver data
   *
   * If the shooting problem has changed after construction, then this function
   * resizes all the data before starting resolve the problem.
   */
  virtual void resizeData();

  /**
   * @brief Compute the feasibility from a given residual vector
   *
   * As in the `computeDynamicFeasibility`, `computeInequalityFeasibility` or
   * `computeEqualityFeasibility`, we can compute the feasibility using
   * different norms (e.g, \f$\ell_\infty\f$ or \f$\ell_1\f$ norms). By default
   * we use the \f$\ell_\infty\f$ norm, however, we can change the type of norm
   * using `set_feasnorm`.
   *
   * @param[in] fs  Vector of residual vectors which we wish to compute the
   * feasibility
   * @return  the residuals' feasibility
   */
  Scalar computeFeasibility(const std::vector<VectorXs>& fs);

  /**
   * @brief Compute the dynamic feasibility
   * \f$\|\mathbf{f}_{\mathbf{s}}\|_{\infty,1}\f$ for the current guess
   * \f$(\mathbf{x}^k,\mathbf{u}^k)\f$
   *
   * The feasibility can be computed using different norms (e.g,
   * \f$\ell_\infty\f$ or \f$\ell_1\f$ norms). By default we use the
   * \f$\ell_\infty\f$ norm, however, we can change the type of norm using
   * `set_feasnorm`. Note that \f$\mathbf{f}_{\mathbf{s}}\f$ are the gaps on the
   * dynamics, which are computed at each node as
   * \f$\mathbf{x}^{'}-\mathbf{f}(\mathbf{x},\mathbf{u})\f$.
   */
  Scalar computeDynamicFeasibility();

  /**
   * @brief Compute the state box-constraint feasibility from a given state
   * trajectory
   *
   * As in the `computeDynamicFeasibility`, `computeInequalityFeasibility` or
   * `computeEqualityFeasibility`, we can compute the feasibility using
   * different norms (e.g, \f$\ell_\infty\f$ or \f$\ell_1\f$ norms). By default
   * we use the \f$\ell_\infty\f$ norm, however, we can change the type of norm
   * using `set_feasnorm`.
   *
   * @param[in] xs  Vector state trajectory
   * feasibility
   * @return  the state trajectory's feasibility
   */
  Scalar computeStateFeasibility(const std::vector<VectorXs>& xs);

  /**
   * @brief Compute the control box-constraint feasibility from a given control
   * trajectory
   *
   * As in the `computeDynamicFeasibility`, `computeInequalityFeasibility` or
   * `computeEqualityFeasibility`, we can compute the feasibility using
   * different norms (e.g, \f$\ell_\infty\f$ or \f$\ell_1\f$ norms). By default
   * we use the \f$\ell_\infty\f$ norm, however, we can change the type of norm
   * using `set_feasnorm`.
   *
   * @param[in] us  Vector control trajectory
   * feasibility
   * @return  the control trajectory's feasibility
   */
  Scalar computeControlFeasibility(const std::vector<VectorXs>& us);

  /**
   * @brief Compute the feasibility of the inequality constraints for the
   * current guess
   *
   * The feasibility can be computed using different norms (e.g,
   * \f$\ell_\infty\f$ or \f$\ell_1\f$ norms). By default we use the
   * \f$\ell_\infty\f$ norm, however, we can change the type of norm using
   * `set_feasnorm`.
   */
  Scalar computeInequalityFeasibility();

  /**
   * @brief Compute the feasibility of the equality constraints for the current
   * guess
   *
   * The feasibility can be computed using different norms (e.g,
   * \f$\ell_\infty\f$ or \f$\ell_1\f$ norms). By default we use the
   * \f$\ell_\infty\f$ norm, however, we can change the type of norm using
   * `set_feasnorm`.
   */
  Scalar computeEqualityFeasibility();

  /**
   * @brief Set a list of callback functions using for the solver diagnostic
   *
   * Each iteration, the solver calls these set of functions in order to allowed
   * user the diagnostic of its performance.
   *
   * @param  callbacks  set of callback functions
   */
  void setCallbacks(
      const std::vector<std::shared_ptr<CallbackAbstract>>& callbacks);

  /**
   * @brief Return the list of callback functions using for diagnostic
   */
  const std::vector<std::shared_ptr<CallbackAbstract>>& getCallbacks() const;

  /**
   * @brief Return the shooting problem
   */
  const std::shared_ptr<ShootingProblem>& get_problem() const;

  /**
   * @brief Return the set of step lengths using by the line-search procedure
   */
  const std::vector<Scalar>& get_alphas() const;

  /**
   * @brief Return the state trajectory \f$\mathbf{x}_s\f$
   */
  const std::vector<VectorXs>& get_xs() const;

  /**
   * @brief Return the control trajectory \f$\mathbf{u}_s\f$
   */
  const std::vector<VectorXs>& get_us() const;

  /**
   * @brief Return the dynamic infeasibility \f$\mathbf{f}_{s}\f$
   */
  const std::vector<VectorXs>& get_fs() const;

  /**
   * @brief Return the linear update in \f$\delta\mathbf{x}_s\f$
   */
  const std::vector<VectorXs>& get_dxs() const;

  /**
   * @brief Return the feedforward gains \f$\delta\mathbf{u}_s\f$
   */
  const std::vector<VectorXs>& get_dus() const;

  /**
   * @brief Return the trial state trajectory \f$\mathbf{x}_s\f$
   */
  const std::vector<VectorXs>& get_xs_try() const;

  /**
   * @brief Return the trial control trajectory \f$\mathbf{u}_s\f$
   */
  const std::vector<VectorXs>& get_us_try() const;

  /**
   * @brief Return the trail dynamic infeasibility \f$\mathbf{f}_{s}\f$
   */
  const std::vector<VectorXs>& get_fs_try() const;

  /**
   * @brief Return the feasibility status of the
   * \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ trajectory
   */
  bool get_is_feasible() const;

  /**
   * @brief Return the cost for the current guess
   */
  Scalar get_cost() const;

  /**
   * @brief Return the merit for the current guess
   */
  Scalar get_merit() const;

  /**
   * @brief Return the stopping-criteria value computed by `stoppingCriteria()`
   */
  Scalar get_stop() const;

  /**
   * @brief Return the constant, linear, and quadratic terms of the expected
   * improvement
   */
  const Vector3s& get_DV() const;

  /**
   * @brief Return the reduction in the cost function \f$\Delta V\f$
   */
  Scalar get_dV() const;

  /**
   * @brief Return the reduction in the merit function \f$\Delta\Phi\f$
   */
  Scalar get_dPhi() const;

  /**
   * @brief Return the expected reduction in the cost function \f$\Delta
   * V_{exp}\f$
   */
  Scalar get_dVexp() const;

  /**
   * @brief Return the expected reduction in the merit function
   * \f$\Delta\Phi_{exp}\f$
   */
  Scalar get_dPhiexp() const;

  /**
   * @brief Return the reduction in the feasibility
   */
  Scalar get_dfeas() const;

  /**
   * @brief Return the total feasibility for the current guess
   */
  Scalar get_feas() const;

  /**
   * @brief Return the dynamic feasibility for the current guess
   */
  Scalar get_ffeas() const;

  /**
   * @brief Return the inequality feasibility for the current guess
   */
  Scalar get_gfeas() const;

  /**
   * @brief Return the equality feasibility for the current guess
   */
  Scalar get_hfeas() const;

  /**
   * @brief Return the dynamic feasibility for the current step length
   */
  Scalar get_ffeas_try() const;

  /**
   * @brief Return the inequality feasibility for the current step length
   */
  Scalar get_gfeas_try() const;

  /**
   * @brief Return the equality feasibility for the current step length
   */
  Scalar get_hfeas_try() const;

  /**
   * @brief Return the primal-variable regularization
   */
  Scalar get_preg() const;

  /**
   * @brief Return the dual-variable regularization
   */
  Scalar get_dreg() const;

  DEPRECATED(
      "Use get_preg for primal-variable regularization",
      Scalar get_xreg() const { return preg_; })
  DEPRECATED(
      "Use get_preg for primal-variable regularization",
      Scalar get_ureg() const { return preg_; })

  /**
   * @brief Return the minimum regularization value
   */
  Scalar get_reg_min() const;

  /**
   * @brief Return the maximum regularization value
   */
  Scalar get_reg_max() const;

  /**
   * @brief Return the step length \f$\alpha\f$
   */
  Scalar get_steplength() const;

  /**
   * @brief Return the threshold used for accepting a step
   */
  Scalar get_th_acceptstep() const;

  /**
   * @brief Return the tolerance for stopping the algorithm
   */
  Scalar get_th_stop() const;

  /**
   * @brief Return the threshold for accepting a gap as non-zero
   */
  DEPRECATED(
      "Do not use this threshold. It is not needed by our solvers",
      Scalar get_th_gaptol() const { return th_gaptol_; })

  /**
   * @brief Return the type of norm used to evaluate the dynamic and constraints
   * feasibility
   */
  FeasibilityNorm get_feasnorm() const;

  /**
   * @brief Return the number of iterations performed by the solver
   */
  std::size_t get_iter() const;

  /**
   * @brief Modify the set of step lengths using by the line-search procedure
   */
  void set_alphas(const std::vector<Scalar>& alphas);

  /**
   * @brief Modify the state trajectory \f$\mathbf{x}_s\f$
   */
  void set_xs(const std::vector<VectorXs>& xs);

  /**
   * @brief Modify the control trajectory \f$\mathbf{u}_s\f$
   */
  void set_us(const std::vector<VectorXs>& us);

  /**
   * @brief Modify the primal-variable regularization value
   */
  void set_preg(const Scalar preg);

  /**
   * @brief Modify the dual-variable regularization value
   */
  void set_dreg(const Scalar dreg);

  /**
   * @brief Modify the minimum regularization value
   */
  void set_reg_min(const Scalar regmin);

  /**
   * @brief Modify the maximum regularization value
   */
  void set_reg_max(const Scalar regmax);

  DEPRECATED(
      "Use set_preg for primal-variable regularization",
      void set_xreg(const Scalar xreg) {
        if (xreg < Scalar(0.)) {
          throw_pretty(
              "Invalid argument: " << "xreg value has to be positive.");
        }
        xreg_ = xreg;
        preg_ = xreg;
      })
  DEPRECATED(
      "Use set_preg for primal-variable regularization",
      void set_ureg(const Scalar ureg) {
        if (ureg < Scalar(0.)) {
          throw_pretty(
              "Invalid argument: " << "ureg value has to be positive.");
        }
        ureg_ = ureg;
        preg_ = ureg;
      })

  /**
   * @brief Modify the threshold used for accepting step
   */
  void set_th_acceptstep(const Scalar th_acceptstep);

  /**
   * @brief Modify the tolerance for stopping the algorithm
   */
  void set_th_stop(const Scalar th_stop);

  /**
   * @brief Modify the threshold for accepting a gap as non-zero
   */
  DEPRECATED("Do not use this threshold. It is not needed by our solvers",
             void set_th_gaptol(const Scalar th_gaptol);)

  /**
   * @brief Modify the current norm used for computed the dynamic and constraint
   * feasibility
   */
  void set_feasnorm(const FeasibilityNorm feas_norm);

 protected:
  /**
   * @brief Allocate all the basic data needed in solvers
   */
  void allocateData();

  /**
   * @brief Resize data associated with the running models of the shooting
   * problem.
   *
   * This function is invoked by `resizeData()` whenever the horizon or the
   * running action models change. Derived classes can override it to adjust
   * their own storage before the solver resumes.
   */
  virtual void resizeRunningData();

  /**
   * @brief Resize data associated with the terminal model of the shooting
   * problem.
   *
   * This function is invoked by `resizeData()` whenever the terminal model
   * changes. Derived classes can override it to resize custom storage that
   * depends on the terminal constraints or cost.
   */
  virtual void resizeTerminalData();

  std::shared_ptr<ShootingProblem> problem_;  //!< optimal control problem
  std::vector<std::shared_ptr<CallbackAbstract>>
      callbacks_;  //!< Callback functions
  std::vector<Scalar>
      alphas_;  //!< Set of step lengths using by the line-search procedure
  Scalar th_acceptstep_;  //!< Threshold used for accepting step
  Scalar th_stop_;        //!< Tolerance for stopping the algorithm
  DEPRECATED("Do not use this threshold. It is not needed by our solvers",
             Scalar th_gaptol_;)   //!< Threshold limit to check non-zero gaps
  enum FeasibilityNorm feasnorm_;  //!< Type of norm used to evaluate the
                                   //!< dynamics and constraints feasibility

  // allocate data
  Scalar dImpr_;  //!< Reduction in the iteration improvement (i.e., maximum
                  //!< between cost and merit values)
  std::vector<VectorXs> xs_;  //!< State trajectory
  std::vector<VectorXs> us_;  //!< Control trajectory
  std::vector<VectorXs> fs_;  //!< Dynamics gaps in each node
  std::vector<VectorXs>
      xs_try_;  //!< State trajectory computed by line-search procedure
  std::vector<VectorXs>
      us_try_;  //!< Control trajectory computed by line-search procedure
  std::vector<VectorXs> fs_try_;  //!< Dynamics gaps in each node computed by
                                  //!< the line-search procedure
  std::vector<VectorXs> dxs_;     //!< Linear state direction (size T + 1)
  std::vector<VectorXs> dus_;     //!< Linear control direction (size T)
  bool is_feasible_;   //!< Label that indicates is the iteration is feasible
  bool was_feasible_;  //!< Label that indicates in the previous iterate was
                       //!< feasible
  Scalar cost_;        //!< Cost for the current guess
  Scalar cost_try_;    //!< Total cost computed by line-search procedure
  Scalar merit_;       //!< Merit for the current guess
  Scalar stop_;        //!< Value computed by `stoppingCriteria()`
  std::size_t iter_;   //!< Number of iteration performed by the solver
  Vector3s DV_;        //!< LQ approximation of the expected improvement
  Scalar dV_;    //!< Reduction in the cost function computed by `tryStep()`
  Scalar dPhi_;  //!< Reduction in the merit function computed by `tryStep()`
  Scalar dVexp_full_;  //!< Expected reduction in the cost function for a full
                       //!< step length
  Scalar dVexp_;  //!< Expected reduction in the cost function for the selected
                  //!< step length
  Scalar dPhiexp_;  //!< Expected reduction in the merit function for the
                    //!< selected step length
  Scalar dfeas_;    //!< Reduction in the feasibility
  Scalar feas_;     //!< Total feasibility for the current guess
  Scalar
      ffeas_;  //!< Feasibility of the dynamic constraints for the current guess
  Scalar gfeas_;  //!< Feasibility of the inequality constraints for the current
                  //!< guess
  Scalar hfeas_;  //!< Feasibility of the equality constraints for the current
                  //!< guess
  Scalar ffeas_try_;  //!< Feasibility of the dynamic constraints evaluated for
                      //!< the current step length
  Scalar gfeas_try_;  //!< Feasibility of the inequality constraints evaluated
                      //!< for the current step length
  Scalar hfeas_try_;  //!< Feasibility of the equality constraints evaluated for
                      //!< the current step length
  Scalar tmp_feas_;   //!< Temporal variables used for computed the feasibility
  Scalar preg_;       //!< Current primal-variable regularization value
  Scalar dreg_;       //!< Current dual-variable regularization value
  DEPRECATED("Use preg_ for primal-variable regularization",
             Scalar xreg_;)  //!< Current state regularization value
  DEPRECATED("Use dreg_ for primal-variable regularization",
             Scalar ureg_;)      //!< Current control regularization values
  Scalar reg_min_;               //!< Minimum allowed regularization value
  Scalar reg_max_;               //!< Maximum allowed regularization value
  Scalar steplength_;            //!< Current applied step length
  std::vector<VectorXs> g_adj_;  //!< Adjusted inequality bound
  std::vector<VectorXs> x_adj_;  //!< Adjusted state bound
  std::vector<VectorXs> u_adj_;  //!< Adjusted control bound
  std::size_t nh_T_;             //!< Dimension of terminal equality constraints
  std::size_t ng_T_;  //!< Dimension of termianl inequality constraints

  bool acceptstep_;
  bool recalcdir_;

 private:
  SolverAbstractTpl() : problem_(nullptr) {}
};

class CallbackBase {
 public:
  virtual ~CallbackBase() = default;

  CROCODDYL_BASE_CAST(CallbackBase, CallbackAbstractTpl)
};

/**
 * @brief Abstract class for solver callbacks
 *
 * A callback is used to diagnostic the behaviour of our solver in each
 * iteration of it. For instance, it can be used to print values, record data or
 * display motions.
 */
template <typename _Scalar>
class CallbackAbstractTpl : public CallbackBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;

  /**
   * @brief Initialize the callback function
   */
  CallbackAbstractTpl() {}
  virtual ~CallbackAbstractTpl() = default;

  /**
   * @brief Run the callback function given a solver
   *
   * @param[in]  solver solver to be diagnostic
   */
  virtual void operator()(SolverAbstract& solver) = 0;
};

template <typename Scalar>
bool raiseIfNaN(const Scalar value);

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/solver-base.hxx"

CROCODDYL_DISABLE_WARNING_DEPRECATED

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SolverAbstractTpl)

CROCODDYL_ENABLE_WARNING_DEPRECATED

#endif  // CROCODDYL_CORE_SOLVER_BASE_HPP_
