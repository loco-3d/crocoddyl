///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2023, IRI: CSIC-UPC, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef __CROCODDYL_CORE_SOLVERS_IPOPT_IPOPT_IFACE_HPP__
#define __CROCODDYL_CORE_SOLVERS_IPOPT_IPOPT_IFACE_HPP__

#define HAVE_CSTDDEF
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF

#include "crocoddyl/core/optctrl/shooting.hpp"

namespace crocoddyl {

struct IpoptInterfaceData;

/**
 * @brief Class for interfacing a crocoddyl::ShootingProblem with IPOPT
 *
 * This class implements the pure virtual functions from Ipopt::TNLP to solve
 * the optimal control problem in `problem_` using a multiple shooting approach.
 *
 * Ipopt considers its decision variables `x` to belong to the Euclidean space.
 * However, Crocoddyl states could lie in a manifold. To ensure that the
 * solution of Ipopt lies in the manifold of the state, we perform the
 * optimization in the tangent space of a given initial state. Finally we
 * retract the Ipopt solution to the manifold. That is:
 *  * \f[
 * \begin{aligned}
 * \mathbf{x}^* = \mathbf{x}^0 \oplus \mathbf{\Delta x}^*
 * \end{aligned}
 * \f]
 *
 * where \f$\mathbf{x}^*\f$ is the final solution, \f$\mathbf{x}^0\f$ is the
 * initial guess and \f$\mathbf{\Delta x}^*\f$ is the Ipopt solution in the
 * tangent space of \f$\mathbf{x}_0\f$. Due to this procedure, the computation
 * of the cost function, the dynamic constraint as well as their corresponding
 * derivatives should be properly modified.
 *
 * The Ipopt decision vector is built as follows: \f$x = [ \mathbf{\Delta
 * x}_0^\top, \mathbf{u}_0^\top, \mathbf{\Delta x}_1^\top, \mathbf{u}_1^\top,
 * \dots, \mathbf{\Delta x}_N^\top ]\f$
 *
 * Dynamic constraints are posed as: \f$(\mathbf{x}^0_{k+1} \oplus
 * \mathbf{\Delta x}_{k+1}) \ominus \mathbf{f}(\mathbf{x}_{k}^0 \oplus
 * \mathbf{\Delta x}_{k}, \mathbf{u}_k) = \mathbf{0}\f$
 *
 * Initial condition: \f$ \mathbf{x}(0) \ominus (\mathbf{x}_{k}^0 \oplus
 * \mathbf{\Delta x}_{k}) = \mathbf{0}\f$
 *
 * Documentation of the methods has been extracted from Ipopt::TNLP.hpp file
 *
 *  \sa `get_nlp_info()`, `get_bounds_info()`, `eval_f()`, `eval_g()`,
 * `eval_grad_f()`, `eval_jac_g()`, `eval_h()`
 */

class IpoptInterface : public Ipopt::TNLP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the Ipopt interface
   *
   * @param[in] problem  Crocoddyl shooting problem
   */
  IpoptInterface(const std::shared_ptr<crocoddyl::ShootingProblem>& problem);

  virtual ~IpoptInterface();

  /**
   * @brief  Methods to gather information about the NLP
   *
   * %Ipopt uses this information when allocating the arrays that it will later
   * ask you to fill with values. Be careful in this method since incorrect
   * values will cause memory bugs which may be very difficult to find.
   * @param[out] n            Storage for the number of variables \f$x\f$
   * @param[out] m            Storage for the number of constraints \f$g(x)\f$
   * @param[out] nnz_jac_g    Storage for the number of nonzero entries in the
   * Jacobian
   * @param[out] nnz_h_lag    Storage for the number of nonzero entries in the
   * Hessian
   * @param[out] index_style  Storage for the index style the numbering style
   * used for row/col entries in the sparse matrix format
   */
  virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                            Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                            IndexStyleEnum& index_style);

  /**
   * @brief Method to request bounds on the variables and constraints.
   T
   * @param[in]  n    Number of variables \f$x\f$ in the problem
   * @param[out] x_l  Lower bounds \f$x^L\f$ for the variables \f$x\f$
   * @param[out] x_u  Upper bounds \f$x^U\f$ for the variables \f$x\f$
   * @param[in]  m    Number of constraints \f$g(x)\f$ in the problem
   * @param[out] g_l  Lower bounds \f$g^L\f$ for the constraints \f$g(x)\f$
   * @param[out] g_u  Upper bounds \f$g^U\f$ for the constraints \f$g(x)\f$
   *
   * @return true if success, false otherwise.
   *
   * The values of `n` and `m` that were specified in
   IpoptInterface::get_nlp_info are passed
   * here for debug checking. Setting a lower bound to a value less than or
   * equal to the value of the option \ref OPT_nlp_lower_bound_inf
   "nlp_lower_bound_inf"
   * will cause %Ipopt to assume no lower bound. Likewise, specifying the upper
   bound above or
   * equal to the value of the option \ref OPT_nlp_upper_bound_inf
   "nlp_upper_bound_inf"
   * will cause %Ipopt to assume no upper bound. These options are set to
   -10<sup>19</sup> and
   * 10<sup>19</sup>, respectively, by default, but may be modified by changing
   these
   * options.
   */
  virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l,
                               Ipopt::Number* x_u, Ipopt::Index m,
                               Ipopt::Number* g_l, Ipopt::Number* g_u);

  /**
   * \brief Method to request the starting point before iterating.
   *
   * @param[in]  n            Number of variables \f$x\f$ in the problem; it
   * will have the same value that was specified in
   * `IpoptInterface::get_nlp_info`
   * @param[in]  init_x       If true, this method must provide an initial value
   * for \f$x\f$
   * @param[out] x            Initial values for the primal variables \f$x\f$
   * @param[in]  init_z       If true, this method must provide an initial value
   * for the bound multipliers \f$z^L\f$ and \f$z^U\f$
   * @param[out] z_L          Initial values for the bound multipliers \f$z^L\f$
   * @param[out] z_U          Initial values for the bound multipliers \f$z^U\f$
   * @param[in]  m            Number of constraints \f$g(x)\f$ in the problem;
   * it will have the same value that was specified in
   * `IpoptInterface::get_nlp_info`
   * @param[in]  init_lambda  If true, this method must provide an initial value
   * for the constraint multipliers \f$\lambda\f$
   * @param[out] lambda       Initial values for the constraint multipliers,
   * \f$\lambda\f$
   *
   * @return true if success, false otherwise.
   *
   * The boolean variables indicate whether the algorithm requires to have x,
   * z_L/z_u, and lambda initialized, respectively. If, for some reason, the
   * algorithm requires initializations that cannot be provided, false should be
   * returned and %Ipopt will stop. The default options only require initial
   * values for the primal variables \f$x\f$.
   *
   * Note, that the initial values for bound multiplier components for absent
   * bounds (\f$x^L_i=-\infty\f$ or \f$x^U_i=\infty\f$) are ignored.
   */
  // [TNLP_get_starting_point]
  virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                                  bool init_z, Ipopt::Number* z_L,
                                  Ipopt::Number* z_U, Ipopt::Index m,
                                  bool init_lambda, Ipopt::Number* lambda);

  /**
   * @brief Method to request the value of the objective function.
   *
   * @param[in] n           Number of variables \f$x\f$ in the problem; it will
   * have the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[in] x           Values for the primal variables \f$x\f$ at which the
   * objective function \f$f(x)\f$ is to be evaluated
   * @param[in] new_x       False if any evaluation method (`eval_*`) was
   * previously called with the same values in x, true otherwise. This can be
   * helpful when users have efficient implementations that calculate multiple
   * outputs at once. %Ipopt internally caches results from the TNLP and
   *                        generally, this flag can be ignored.
   * @param[out] obj_value  Storage for the value of the objective function
   * \f$f(x)\f$
   *
   * @return true if success, false otherwise.
   */
  virtual bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                      Ipopt::Number& obj_value);

  /**
   * @brief Method to request the gradient of the objective function.
   *
   * @param[in] n        Number of variables \f$x\f$ in the problem; it will
   * have the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[in] x        Values for the primal variables \f$x\f$ at which the
   * gradient \f$\nabla f(x)\f$ is to be evaluated
   * @param[in] new_x    False if any evaluation method (`eval_*`) was
   * previously called with the same values in x, true otherwise; see also
   * `IpoptInterface::eval_f`
   * @param[out] grad_f  Array to store values of the gradient of the objective
   * function \f$\nabla f(x)\f$. The gradient array is in the same order as the
   * \f$x\f$ variables (i.e., the gradient of the objective with respect to
   * `x[2]` should be put in `grad_f[2]`).
   *
   * @return true if success, false otherwise.
   */
  virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                           Ipopt::Number* grad_f);

  /**
   * @brief Method to request the constraint values.
   *
   * @param[in] n      Number of variables \f$x\f$ in the problem; it will have
   * the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[in] x      Values for the primal variables \f$x\f$ at which the
   * constraint functions \f$g(x)\f$ are to be evaluated
   * @param[in] new_x  False if any evaluation method (`eval_*`) was previously
   * called with the same values in x, true otherwise; see also
   * `IpoptInterface::eval_f`
   * @param[in] m      Number of constraints \f$g(x)\f$ in the problem; it will
   * have the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[out] g     Array to store constraint function values \f$g(x)\f$, do
   * not add or subtract the bound values \f$g^L\f$ or \f$g^U\f$.
   *
   * @return true if success, false otherwise.
   */
  virtual bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                      Ipopt::Index m, Ipopt::Number* g);

  /**
   * @brief Method to request either the sparsity structure or the values of the
   * Jacobian of the constraints.
   *
   * The Jacobian is the matrix of derivatives where the derivative of
   * constraint function \f$g_i\f$ with respect to variable \f$x_j\f$ is placed
   * in row \f$i\f$ and column \f$j\f$. See \ref TRIPLET for a discussion of the
   * sparse matrix format used in this method.
   *
   * @param[in] n         Number of variables \f$x\f$ in the problem; it will
   * have the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[in] x         First call: NULL; later calls: the values for the
   * primal variables \f$x\f$ at which the constraint Jacobian \f$\nabla
   * g(x)^T\f$ is to be evaluated
   * @param[in] new_x     False if any evaluation method (`eval_*`) was
   * previously called with the same values in x, true otherwise; see also
   * `IpoptInterface::eval_f`
   * @param[in] m         Number of constraints \f$g(x)\f$ in the problem; it
   * will have the same value that was specified in
   * `IpoptInterface::get_nlp_info`
   * @param[in] nele_jac  Number of nonzero elements in the Jacobian; it will
   * have the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[out] iRow     First call: array of length `nele_jac` to store the
   * row indices of entries in the Jacobian f the constraints; later calls: NULL
   * @param[out] jCol     First call: array of length `nele_jac` to store the
   * column indices of entries in the acobian of the constraints; later calls:
   * NULL
   * @param[out] values   First call: NULL; later calls: array of length
   * nele_jac to store the values of the entries in the Jacobian of the
   * constraints
   *
   * @return true if success, false otherwise.
   *
   * @note The arrays iRow and jCol only need to be filled once. If the iRow and
   * jCol arguments are not NULL (first call to this function), then %Ipopt
   * expects that the sparsity structure of the Jacobian (the row and column
   * indices only) are written into iRow and jCol. At this call, the arguments
   * `x` and `values` will be NULL. If the arguments `x` and `values` are not
   * NULL, then %Ipopt expects that the value of the Jacobian as calculated from
   * array `x` is stored in array `values` (using the same order as used when
   * specifying the sparsity structure). At this call, the arguments `iRow` and
   * `jCol` will be NULL.
   */
  virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                          Ipopt::Index m, Ipopt::Index nele_jac,
                          Ipopt::Index* iRow, Ipopt::Index* jCol,
                          Ipopt::Number* values);

  /**
   * @brief Method to request either the sparsity structure or the values of the
   * Hessian of the Lagrangian.
   *
   * The Hessian matrix that %Ipopt uses is
   * \f[ \sigma_f \nabla^2 f(x_k) + \sum_{i=1}^m\lambda_i\nabla^2 g_i(x_k) \f]
   * for the given values for \f$x\f$, \f$\sigma_f\f$, and \f$\lambda\f$.
   * See \ref TRIPLET for a discussion of the sparse matrix format used in this
   * method.
   *
   * @param[in] n           Number of variables \f$x\f$ in the problem; it will
   * have the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[in] x           First call: NULL; later calls: the values for the
   * primal variables \f$x\f$ at which the Hessian is to be evaluated
   * @param[in] new_x       False if any evaluation method (`eval_*`) was
   * previously called with the same values in x, true otherwise; see also
   * IpoptInterface::eval_f
   * @param[in] obj_factor  Factor \f$\sigma_f\f$ in front of the objective term
   * in the Hessian
   * @param[in] m           Number of constraints \f$g(x)\f$ in the problem; it
   * will have the same value that was specified in
   * `IpoptInterface::get_nlp_info`
   * @param[in] lambda      Values for the constraint multipliers \f$\lambda\f$
   * at which the Hessian is to be evaluated
   * @param[in] new_lambda  False if any evaluation method was previously called
   * with the same values in lambda, true otherwise
   * @param[in] nele_hess   Number of nonzero elements in the Hessian; it will
   * have the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[out] iRow       First call: array of length nele_hess to store the
   * row indices of entries in the Hessian; later calls: NULL
   * @param[out] jCol       First call: array of length nele_hess to store the
   * column indices of entries in the Hessian; later calls: NULL
   * @param[out] values     First call: NULL; later calls: array of length
   * nele_hess to store the values of the entries in the Hessian
   *
   * @return true if success, false otherwise.
   *
   * @note The arrays iRow and jCol only need to be filled once. If the iRow and
   * jCol arguments are not NULL (first call to this function), then %Ipopt
   * expects that the sparsity structure of the Hessian (the row and column
   * indices only) are written into iRow and jCol. At this call, the arguments
   * `x`, `lambda`, and `values` will be NULL. If the arguments `x`, `lambda`,
   * and `values` are not NULL, then %Ipopt expects that the value of the
   * Hessian as calculated from arrays `x` and `lambda` are stored in array
   * `values` (using the same order as used when specifying the sparsity
   * structure). At this call, the arguments `iRow` and `jCol` will be NULL.
   *
   * @attention As this matrix is symmetric, %Ipopt expects that only the lower
   * diagonal entries are specified.
   *
   * A default implementation is provided, in case the user wants to set
   * quasi-Newton approximations to estimate the second derivatives and doesn't
   * not need to implement this method.
   */
  virtual bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                      Ipopt::Number obj_factor, Ipopt::Index m,
                      const Ipopt::Number* lambda, bool new_lambda,
                      Ipopt::Index nele_hess, Ipopt::Index* iRow,
                      Ipopt::Index* jCol, Ipopt::Number* values);

  /**
   * @brief This method is called when the algorithm has finished (successfully
   * or not) so the TNLP can digest the outcome, e.g., store/write the solution,
   * if any.
   *
   * @param[in] status @parblock gives the status of the algorithm
   *   - SUCCESS: Algorithm terminated successfully at a locally optimal
   *     point, satisfying the convergence tolerances (can be specified
   *     by options).
   *   - MAXITER_EXCEEDED: Maximum number of iterations exceeded (can be
   * specified by an option).
   *   - CPUTIME_EXCEEDED: Maximum number of CPU seconds exceeded (can be
   * specified by an option).
   *   - STOP_AT_TINY_STEP: Algorithm proceeds with very little progress.
   *   - STOP_AT_ACCEPTABLE_POINT: Algorithm stopped at a point that was
   * converged, not to "desired" tolerances, but to "acceptable" tolerances (see
   * the acceptable-... options).
   *   - LOCAL_INFEASIBILITY: Algorithm converged to a point of local
   * infeasibility. Problem may be infeasible.
   *   - USER_REQUESTED_STOP: The user call-back function
   * IpoptInterface::intermediate_callback returned false, i.e., the user code
   * requested a premature termination of the optimization.
   *   - DIVERGING_ITERATES: It seems that the iterates diverge.
   *   - RESTORATION_FAILURE: Restoration phase failed, algorithm doesn't know
   * how to proceed.
   *   - ERROR_IN_STEP_COMPUTATION: An unrecoverable error occurred while %Ipopt
   * tried to compute the search direction.
   *   - INVALID_NUMBER_DETECTED: Algorithm received an invalid number (such as
   * NaN or Inf) from the NLP; see also option check_derivatives_for_nan_inf).
   *   - INTERNAL_ERROR: An unknown internal error occurred.
   * @endparblock
   * @param[in] n           Number of variables \f$x\f$ in the problem; it will
   * have the same value that was specified in `IpoptInterface::get_nlp_info`
   * @param[in] x           Final values for the primal variables
   * @param[in] z_L         Final values for the lower bound multipliers
   * @param[in] z_U         Final values for the upper bound multipliers
   * @param[in] m           Number of constraints \f$g(x)\f$ in the problem; it
   * will have the same value that was specified in
   * `IpoptInterface::get_nlp_info`
   * @param[in] g           Final values of the constraint functions
   * @param[in] lambda      Final values of the constraint multipliers
   * @param[in] obj_value   Final value of the objective function
   * @param[in] ip_data     Provided for expert users
   * @param[in] ip_cq       Provided for expert users
   */
  virtual void finalize_solution(
      Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* x,
      const Ipopt::Number* z_L, const Ipopt::Number* z_U, Ipopt::Index m,
      const Ipopt::Number* g, const Ipopt::Number* lambda,
      Ipopt::Number obj_value, const Ipopt::IpoptData* ip_data,
      Ipopt::IpoptCalculatedQuantities* ip_cq);

  /**
   * @brief Intermediate Callback method for the user.
   *
   * This method is called once per iteration (during the convergence check),
   * and can be used to obtain information about the optimization status while
   * %Ipopt solves the problem, and also to request a premature termination.
   *
   * The information provided by the entities in the argument list correspond to
   * what %Ipopt prints in the iteration summary (see also \ref OUTPUT). Further
   * information can be obtained from the ip_data and ip_cq objects. The current
   * iterate and violations of feasibility and optimality can be accessed via
   * the methods IpoptInterface::get_curr_iterate() and
   * IpoptInterface::get_curr_violations(). These methods translate values for
   * the *internal representation* of the problem from `ip_data` and `ip_cq`
   * objects into the TNLP representation.
   *
   * @return If this method returns false, %Ipopt will terminate with the
   * User_Requested_Stop status.
   *
   * It is not required to implement (overload) this method. The default
   * implementation always returns true.
   */
  bool intermediate_callback(
      Ipopt::AlgorithmMode mode, Ipopt::Index iter, Ipopt::Number obj_value,
      Ipopt::Number inf_pr, Ipopt::Number inf_du, Ipopt::Number mu,
      Ipopt::Number d_norm, Ipopt::Number regularization_size,
      Ipopt::Number alpha_du, Ipopt::Number alpha_pr, Ipopt::Index ls_trials,
      const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq);

  /**
   * @brief Create the data structure to store temporary computations
   *
   * @return the IpoptInterface Data
   */
  std::shared_ptr<IpoptInterfaceData> createData(const std::size_t nx,
                                                 const std::size_t ndx,
                                                 const std::size_t nu);

  void resizeData();

  /**
   * @brief Return the total number of optimization variables (states and
   * controls)
   */
  std::size_t get_nvar() const;

  /**
   * @brief Return the total number of constraints in the NLP
   */
  std::size_t get_nconst() const;

  /**
   * @brief Return the state vector
   */
  const std::vector<Eigen::VectorXd>& get_xs() const;

  /**
   * @brief Return the control vector
   */
  const std::vector<Eigen::VectorXd>& get_us() const;

  /**
   * @brief Return the crocoddyl::ShootingProblem to be solved
   */
  const std::shared_ptr<crocoddyl::ShootingProblem>& get_problem() const;

  double get_cost() const;

  /**
   * @brief Modify the state vector
   */
  void set_xs(const std::vector<Eigen::VectorXd>& xs);

  /**
   * @brief Modify the control vector
   */
  void set_us(const std::vector<Eigen::VectorXd>& us);

 private:
  std::shared_ptr<crocoddyl::ShootingProblem>
      problem_;                      //!< Optimal control problem
  std::vector<Eigen::VectorXd> xs_;  //!< Vector of states
  std::vector<Eigen::VectorXd> us_;  //!< Vector of controls
  std::vector<std::size_t> ixu_;     //!< Index of at node i
  std::size_t nvar_;                 //!< Number of NLP variables
  std::size_t nconst_;               //!< Number of the NLP constraints
  std::vector<std::shared_ptr<IpoptInterfaceData>> datas_;  //!< Vector of Datas
  double cost_;                                             //!< Total cost

  IpoptInterface(const IpoptInterface&);

  IpoptInterface& operator=(const IpoptInterface&);
};

struct IpoptInterfaceData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IpoptInterfaceData(const std::size_t nx, const std::size_t ndx,
                     const std::size_t nu)
      : x(nx),
        xnext(nx),
        dx(ndx),
        dxnext(ndx),
        x_diff(ndx),
        u(nu),
        Jint_dx(ndx, ndx),
        Jint_dxnext(ndx, ndx),
        Jdiff_x(ndx, ndx),
        Jdiff_xnext(ndx, ndx),
        Jg_dx(ndx, ndx),
        Jg_dxnext(ndx, ndx),
        Jg_u(ndx, ndx),
        Jg_ic(ndx, ndx),
        FxJint_dx(ndx, ndx),
        Ldx(ndx),
        Ldxdx(ndx, ndx),
        Ldxu(ndx, nu) {
    x.setZero();
    xnext.setZero();
    dx.setZero();
    dxnext.setZero();
    x_diff.setZero();
    u.setZero();
    Jint_dx.setZero();
    Jint_dxnext.setZero();
    Jdiff_x.setZero();
    Jdiff_xnext.setZero();
    Jg_dx.setZero();
    Jg_dxnext.setZero();
    Jg_u.setZero();
    Jg_ic.setZero();
    FxJint_dx.setZero();
    Ldx.setZero();
    Ldxdx.setZero();
    Ldxu.setZero();
  }

  void resize(const std::size_t nx, const std::size_t ndx,
              const std::size_t nu) {
    x.conservativeResize(nx);
    xnext.conservativeResize(nx);
    dx.conservativeResize(ndx);
    dxnext.conservativeResize(ndx);
    x_diff.conservativeResize(ndx);
    u.conservativeResize(nu);
    Jint_dx.conservativeResize(ndx, ndx);
    Jint_dxnext.conservativeResize(ndx, ndx);
    Jdiff_x.conservativeResize(ndx, ndx);
    Jdiff_xnext.conservativeResize(ndx, ndx);
    Jg_dx.conservativeResize(ndx, ndx);
    Jg_dxnext.conservativeResize(ndx, ndx);
    Jg_u.conservativeResize(ndx, ndx);
    Jg_ic.conservativeResize(ndx, ndx);
    FxJint_dx.conservativeResize(ndx, ndx);
    Ldx.conservativeResize(ndx);
    Ldxdx.conservativeResize(ndx, ndx);
    Ldxu.conservativeResize(ndx, nu);
  }

  Eigen::VectorXd x;        //!< Integrated state
  Eigen::VectorXd xnext;    //!< Integrated state at next node
  Eigen::VectorXd dx;       //!< Increment in the tangent space
  Eigen::VectorXd dxnext;   //!< Increment in the tangent space at next node
  Eigen::VectorXd x_diff;   //!< State difference
  Eigen::VectorXd u;        //!< Control
  Eigen::MatrixXd Jint_dx;  //!< Jacobian of the sum operation w.r.t dx
  Eigen::MatrixXd
      Jint_dxnext;  //!< Jacobian of the sum operation w.r.t dx at next node
  Eigen::MatrixXd
      Jdiff_x;  //!< Jacobian of the diff operation w.r.t the first element
  Eigen::MatrixXd Jdiff_xnext;  //!< Jacobian of the diff operation w.r.t the
                                //!< first element at the next node
  Eigen::MatrixXd Jg_dx;        //!< Jacobian of the dynamic constraint w.r.t dx
  Eigen::MatrixXd
      Jg_dxnext;         //!< Jacobian of the dynamic constraint w.r.t dxnext
  Eigen::MatrixXd Jg_u;  //!< Jacobian of the dynamic constraint w.r.t u
  Eigen::MatrixXd
      Jg_ic;  //!< Jacobian of the initial condition constraint w.r.t dx
  Eigen::MatrixXd FxJint_dx;  //!< Intermediate computation needed for Jg_ic
  Eigen::VectorXd Ldx;        //!< Jacobian of the cost w.r.t dx
  Eigen::MatrixXd Ldxdx;      //!< Hessian of the cost w.r.t dxdx
  Eigen::MatrixXd Ldxu;       //!< Hessian of the cost w.r.t dxu
};

}  // namespace crocoddyl

#endif
