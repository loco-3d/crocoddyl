///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

struct BoxQPSolution {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BoxQPSolution() {}
  BoxQPSolution(const Eigen::MatrixXd& Hff_inv, const Eigen::VectorXd& x, const std::vector<size_t>& free_idx,
                const std::vector<size_t>& clamped_idx)
      : Hff_inv(Hff_inv), x(x), free_idx(free_idx), clamped_idx(clamped_idx) {}

  Eigen::MatrixXd Hff_inv;
  Eigen::VectorXd x;
  std::vector<size_t> free_idx;
  std::vector<size_t> clamped_idx;
};

/**
 * @brief This class implements a box QP based on Projected Newton method.
 *
 * We consider a box QP proble of the form:
 * \f{eqnarray*}{
 *   \min_{\mathbf{x}} &= \frac{1}{2}\mathbf{x}^T\mathbf{H}\mathbf{x} + \mathbf{q}^T\mathbf{x} \\
 *   \textrm{subject to} & \hspace{1em} \mathbf{\underline{b}} \leq \mathbf{x} \leq \mathbf{\bar{b}} \\
 * \f}
 * where \f$\mathbf{H}\f$, \f$\mathbf{q}\f$ are the Hessian and gradient of the problem,
 * respectively, \f$\mathbf{\underline{b}}\f$, \f$\mathbf{\bar{b}}\f$ are lower and upper
 * bounds of the decision variable \f$\mathbf{x}\f$.
 *
 * The algorithm procees by iteratively identifying the active bounds, and then
 * performing a projected Newton step in the free sub-space.
 * The projection uses the Hessian of the free sub-space and are computed using
 * Cholesky decomposition.
 * It uses a line search procedure with polynomial step length values in a
 * backtracking fashion.
 * The step are checked using an Armijo condition together L2-norm gradient.
 *
 * For more details about this solver, we encourage you to read the following
 * article:
 * D. P. Bertsekas, "Projected newton methods for optimization problems with
 * simple constraints". SIAM Journal on Control and Optimization.
 */
class BoxQP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BoxQP(const std::size_t nx, std::size_t maxiter = 100, const double th_acceptstep = 0.1, const double th_grad = 1e-9,
        const double reg = 1e-9);
  ~BoxQP();

  const BoxQPSolution& solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& q, const Eigen::VectorXd& lb,
                             const Eigen::VectorXd& ub, const Eigen::VectorXd& xinit);
  const BoxQPSolution& get_solution() const;
  const std::size_t& get_nx() const;
  const std::size_t& get_maxiter() const;
  const double& get_th_acceptstep() const;
  const double& get_th_grad() const;
  const double& get_reg() const;
  const std::vector<double>& get_alphas() const;
  void set_nx(const std::size_t& nx);
  void set_maxiter(const std::size_t& maxiter);
  void set_th_acceptstep(const double& th_acceptstep);
  void set_th_grad(const double& th_grad);
  void set_reg(const double& reg);
  void set_alphas(const std::vector<double>& alphas);

 private:
  std::size_t nx_;
  BoxQPSolution solution_;
  std::size_t maxiter_;
  double th_acceptstep_;
  double th_grad_;
  double reg_;

  double fold_;
  double fnew_;
  std::size_t nf_;
  std::size_t nc_;
  std::vector<double> alphas_;
  Eigen::VectorXd x_;
  Eigen::VectorXd xnew_;
  Eigen::VectorXd g_;
  Eigen::VectorXd dx_;

  Eigen::VectorXd qf_;
  Eigen::VectorXd xf_;
  Eigen::VectorXd xc_;
  Eigen::VectorXd dxf_;
  Eigen::MatrixXd Hff_;
  Eigen::MatrixXd Hfc_;
  Eigen::LLT<Eigen::MatrixXd> Hff_inv_llt_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
