///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <vector>

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Box QP solution
 *
 * It contains the Box QP solution data which consists of
 *  - the inverse of the free space Hessian
 *  - the optimal decision vector
 *  - the indexes for the free space
 *  - the indexes for the clamped (constrained) space
 */
struct BoxQPSolution {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the QP solution structure
   */
  BoxQPSolution() {}

  /**
   * @brief Initialize the QP solution structure
   *
   * @param[in] Hff_inv      Inverse of the free space Hessian
   * @param[in] x            Decision vector
   * @param[in] free_idx     Free space indexes
   * @param[in] clamped_idx  Clamped space indexes
   */
  BoxQPSolution(const Eigen::MatrixXd &Hff_inv, const Eigen::VectorXd &x,
                const std::vector<size_t> &free_idx,
                const std::vector<size_t> &clamped_idx)
      : Hff_inv(Hff_inv), x(x), free_idx(free_idx), clamped_idx(clamped_idx) {}

  Eigen::MatrixXd Hff_inv;         //!< Inverse of the free space Hessian
  Eigen::VectorXd x;               //!< Decision vector
  std::vector<size_t> free_idx;    //!< Free space indexes
  std::vector<size_t> clamped_idx; //!< Clamped space indexes
};

/**
 * @brief This class implements a Box QP solver based on a Projected Newton
 * method.
 *
 * We consider a box QP problem of the form:
 * \f{eqnarray*}{
 *   \min_{\mathbf{x}} &= \frac{1}{2}\mathbf{x}^T\mathbf{H}\mathbf{x} +
 * \mathbf{q}^T\mathbf{x} \\
 *   \textrm{subject to} & \hspace{1em} \mathbf{\underline{b}} \leq \mathbf{x}
 * \leq \mathbf{\bar{b}} \\ \f} where \f$\mathbf{H}\f$, \f$\mathbf{q}\f$ are the
 * Hessian and gradient of the problem, respectively,
 * \f$\mathbf{\underline{b}}\f$, \f$\mathbf{\bar{b}}\f$ are lower and upper
 * bounds of the decision variable \f$\mathbf{x}\f$.
 *
 * The algorithm procees by iteratively identifying the active bounds, and then
 * performing a projected Newton step in the free sub-space.
 * The projection uses the Hessian of the free sub-space and is computed
 * efficiently using a Cholesky decomposition.
 * It uses a line search procedure with polynomial step length values in a
 * backtracking fashion.
 * The steps are checked using an Armijo condition together L2-norm gradient.
 *
 * For more details about this solver, we encourage you to read the following
 * article:
 * \include bertsekas-siam82.bib
 */
class BoxQP {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the Projected-Newton QP for bound constraints
   *
   * @param[in] nx             Dimension of the decision vector
   * @param[in] maxiter        Maximum number of allowed iterations (default
   * 100)
   * @param[in] th_acceptstep  Acceptance step threshold (default 0.1)
   * @param[in] th_grad        Gradient tolerance threshold (default 1e-9)
   * @param[in] reg            Regularization value (default 1e-9)
   */
  BoxQP(const std::size_t nx, std::size_t maxiter = 100,
        const double th_acceptstep = 0.1, const double th_grad = 1e-9,
        const double reg = 1e-9);
  /**
   * @brief Destroy the Projected-Newton QP solver
   */
  ~BoxQP();

  /**
   * @brief Compute the solution of bound-constrained QP based on Newton
   * projection
   *
   * @param[in] H      Hessian (dimension nx * nx)
   * @param[in] q      Gradient (dimension nx)
   * @param[in] lb     Lower bound (dimension nx)
   * @param[in] ub     Upper bound (dimension nx)
   * @param[in] xinit  Initial guess (dimension nx)
   * @return The solution of the problem
   */
  const BoxQPSolution &solve(const Eigen::MatrixXd &H, const Eigen::VectorXd &q,
                             const Eigen::VectorXd &lb,
                             const Eigen::VectorXd &ub,
                             const Eigen::VectorXd &xinit);

  /**
   * @brief Return the stored solution
   */
  const BoxQPSolution &get_solution() const;

  /**
   * @brief Return the decision vector dimension
   */
  const std::size_t &get_nx() const;

  /**
   * @brief Return the maximum allowed number of iterations
   */
  const std::size_t &get_maxiter() const;

  /**
   * @brief Return the acceptance step threshold
   */
  const double &get_th_acceptstep() const;

  /**
   * @brief Return the gradient tolerance threshold
   */
  const double &get_th_grad() const;

  /**
   * @brief Return the regularization value
   */
  const double &get_reg() const;

  /**
   * @brief Return the stack of step lengths using by the line-search procedure
   */
  const std::vector<double> &get_alphas() const;

  /**
   * @brief Modify the decision vector dimension
   */
  void set_nx(const std::size_t &nx);

  /**
   * @brief Modify the maximum allowed number of iterations
   */
  void set_maxiter(const std::size_t &maxiter);

  /**
   * @brief Modify the acceptance step threshold
   */
  void set_th_acceptstep(const double &th_acceptstep);

  /**
   * @brief Modify the gradient tolerance threshold
   */
  void set_th_grad(const double &th_grad);

  /**
   * @brief Modify the regularization value
   */
  void set_reg(const double &reg);

  /**
   * @brief Modify the stack of step lengths using by the line-search procedure
   */
  void set_alphas(const std::vector<double> &alphas);

private:
  std::size_t nx_;         //!< Decision variable dimension
  BoxQPSolution solution_; //!< Solution of the Box QP
  std::size_t maxiter_;    //!< Allowed maximum number of iterations
  double th_acceptstep_;   //!< Threshold used for accepting step
  double
      th_grad_; //!< Tolerance for stopping the algorithm (gradient threshold)
  double reg_;  //!< Current regularization value

  double fold_;    //!< Cost of previous iteration
  double fnew_;    //!< Cost of current iteration
  std::size_t nf_; //!< Free space dimension
  std::size_t nc_; //!< Constrained space dimension
  std::vector<double>
      alphas_; //!< Set of step lengths using by the line-search procedure
  Eigen::VectorXd x_;    //!< Guess of the decision variable
  Eigen::VectorXd xnew_; //!< New decision variable guess
  Eigen::VectorXd g_;    //!< Current gradient
  Eigen::VectorXd dx_;   //!< Current search direction

  Eigen::VectorXd qf_; //!< Current problem gradient in the free subspace
  Eigen::VectorXd xf_; //!< Current decision variable in the free subspace
  Eigen::VectorXd
      xc_; //!< Current decision variable in the constrained subspace
  Eigen::VectorXd dxf_; //!< Search direction in the free subspace
  Eigen::MatrixXd Hff_; //!< Hessian in the free subspace
  Eigen::MatrixXd Hfc_; //!< Hessian in the constrained subspace
  Eigen::LLT<Eigen::MatrixXd> Hff_inv_llt_; //!< Cholesky solver
};

} // namespace crocoddyl

#endif // CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
