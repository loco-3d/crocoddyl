///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_

#include "crocoddyl/core/fwd.hpp"
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
template <typename _Scalar>
struct BoxQPSolutionTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the QP solution structure
   */
  BoxQPSolutionTpl() {}

  /**
   * @brief Initialize the QP solution structure
   *
   * @param[in] Hff_inv      Inverse of the free space Hessian
   * @param[in] x            Decision vector
   * @param[in] free_idx     Free space indexes
   * @param[in] clamped_idx  Clamped space indexes
   */
  BoxQPSolutionTpl(const MatrixXs& Hff_inv, const VectorXs& x,
                   const std::vector<size_t>& free_idx,
                   const std::vector<size_t>& clamped_idx)
      : Hff_inv(Hff_inv), x(x), free_idx(free_idx), clamped_idx(clamped_idx) {}

  /**
   * @brief Cast the BoxQP solution to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return BoxQPSolutionTpl<NewScalar> A BoxQP solution with the new scalar
   * type.
   */
  template <typename NewScalar>
  BoxQPSolutionTpl<NewScalar> cast() const {
    typedef BoxQPSolutionTpl<NewScalar> ReturnType;
    ReturnType ret(Hff_inv.template cast<NewScalar>(),
                   x.template cast<NewScalar>(), free_idx, clamped_idx);
    return ret;
  }

  MatrixXs Hff_inv;                 //!< Inverse of the free space Hessian
  VectorXs x;                       //!< Decision vector
  std::vector<size_t> free_idx;     //!< Free space indexes
  std::vector<size_t> clamped_idx;  //!< Clamped space indexes
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
template <typename _Scalar>
class BoxQPTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef BoxQPSolutionTpl<Scalar> BoxQPSolution;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

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
  BoxQPTpl(const std::size_t nx, const std::size_t maxiter = 100,
           const Scalar th_acceptstep = Scalar(0.1),
           const Scalar th_grad = Scalar(1e-9),
           const Scalar reg = Scalar(1e-9));
  /**
   * @brief Destroy the Projected-Newton QP solver
   */
  ~BoxQPTpl() = default;

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
  const BoxQPSolution& solve(const MatrixXs& H, const VectorXs& q,
                             const VectorXs& lb, const VectorXs& ub,
                             const VectorXs& xinit);

  /**
   * @brief Cast the BoxQP solver to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return BoxQPTpl<NewScalar> A BoxQP solver with the new scalar type.
   */
  template <typename NewScalar>
  BoxQPTpl<NewScalar> cast() const;

  /**
   * @brief Return the stored solution
   */
  const BoxQPSolution& get_solution() const;

  /**
   * @brief Return the decision vector dimension
   */
  std::size_t get_nx() const;

  /**
   * @brief Return the maximum allowed number of iterations
   */
  std::size_t get_maxiter() const;

  /**
   * @brief Return the acceptance step threshold
   */
  Scalar get_th_acceptstep() const;

  /**
   * @brief Return the gradient tolerance threshold
   */
  Scalar get_th_grad() const;

  /**
   * @brief Return the regularization value
   */
  Scalar get_reg() const;

  /**
   * @brief Return the stack of step lengths using by the line-search procedure
   */
  const std::vector<Scalar>& get_alphas() const;

  /**
   * @brief Modify the decision vector dimension
   */
  void set_nx(const std::size_t nx);

  /**
   * @brief Modify the maximum allowed number of iterations
   */
  void set_maxiter(const std::size_t maxiter);

  /**
   * @brief Modify the acceptance step threshold
   */
  void set_th_acceptstep(const Scalar th_acceptstep);

  /**
   * @brief Modify the gradient tolerance threshold
   */
  void set_th_grad(const Scalar th_grad);

  /**
   * @brief Modify the regularization value
   */
  void set_reg(const Scalar reg);

  /**
   * @brief Modify the stack of step lengths using by the line-search procedure
   */
  void set_alphas(const std::vector<Scalar>& alphas);

 private:
  std::size_t nx_;          //!< Decision variable dimension
  BoxQPSolution solution_;  //!< Solution of the Box QP
  std::size_t maxiter_;     //!< Allowed maximum number of iterations
  Scalar th_acceptstep_;    //!< Threshold used for accepting step
  Scalar
      th_grad_;  //!< Tolerance for stopping the algorithm (gradient threshold)
  Scalar reg_;   //!< Current regularization value

  Scalar fold_;     //!< Cost of previous iteration
  Scalar fnew_;     //!< Cost of current iteration
  std::size_t nf_;  //!< Free space dimension
  std::size_t nc_;  //!< Constrained space dimension
  std::vector<Scalar>
      alphas_;     //!< Set of step lengths using by the line-search procedure
  VectorXs x_;     //!< Guess of the decision variable
  VectorXs xnew_;  //!< New decision vector
  VectorXs g_;     //!< Current gradient
  VectorXs dx_;    //!< Current search direction

  VectorXs xo_;  //!< Organized decision
  VectorXs
      dxo_;  //!< Search direction organized by free and constrained subspaces
  VectorXs qo_;  //!< Gradient organized by free and constrained subspaces
  MatrixXs Ho_;  //!< Hessian organized by free and constrained subspaces

  Eigen::LLT<MatrixXs> Hff_inv_llt_;  //!< Cholesky solver
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/solvers/box-qp.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::BoxQPTpl)

#endif  // CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
