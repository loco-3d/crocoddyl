///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_FDDP_HPP_
#define CROCODDYL_CORE_SOLVERS_FDDP_HPP_

#include <Eigen/Cholesky>
#include <vector>

#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {

/**
 * @brief Feasibility-driven Differential Dynamic Programming (FDDP) solver
 *
 * The FDDP solver computes an optimal trajectory and control commands by iterates running `backwardPass()` and
 * `forwardPass()`. The backward pass accepts infeasible guess as described in the `SolverDDP::backwardPass()`.
 * Additionally, the forward pass handles infeasibility simulations that resembles the numerical behaviour of
 * a multiple-shooting formulation, i.e.:
 * \f{eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0 - (1 - \alpha)\mathbf{\bar{f}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k + \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\
 *   \mathbf{\hat{x}}_{k+1} &=& \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k) - (1 -
 * \alpha)\mathbf{\bar{f}}_{k+1}.
 * \f}
 * Note that the forward pass keeps the gaps \f$\mathbf{\bar{f}}_s\f$ open according to the step length \f$\alpha\f$
 * that has been accepted. This solver has shown empirically greater globalization strategy. Additionally, the
 * expected improvement computation considers the gaps in the dynamics:
 * \f{equation}
 *   \Delta J(\alpha) = \Delta_1\alpha + \frac{1}{2}\Delta_2\alpha^2,
 * \f}
 * with
 * \f{eqnarray}
 *   \Delta_1 = \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k} +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k} -
 *   V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
 *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k + \mathbf{\bar{f}}_k^\top(2 V_{\mathbf{xx}_k}\mathbf{x}_k
 * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). \f}
 *
 * For more details about the feasibility-driven differential dynamic programming algorithm see:
 * \include mastalli-icra20.bib
 *
 * \sa `backwardPass()`, `forwardPass()`, `expectedImprovement()` and `updateExpectedImprovement()`
 */
class SolverFDDP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the FDDP solver
   */
  explicit SolverFDDP(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverFDDP();

  virtual bool solve(const crocoddyl::aligned_vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const crocoddyl::aligned_vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR,
                     const std::size_t maxiter = 100, const bool is_feasible = false, const double regInit = 1e-9);

  /**
   * @copybrief SolverAbstract::expectedImprovement
   *
   * This function requires to first run `updateExpectedImprovement()`. The expected improvement computation considers
   * the gaps in the dynamics: \f{equation} \Delta J(\alpha) = \Delta_1\alpha + \frac{1}{2}\Delta_2\alpha^2, \f} with
   * \f{eqnarray}
   *   \Delta_1 = \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k} +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k}
   * - V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
   *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k + \mathbf{\bar{f}}_k^\top(2
   * V_{\mathbf{xx}_k}\mathbf{x}_k
   * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). \f}
   */
  virtual const Eigen::Vector2d& expectedImprovement();

  /**
   * @brief Update internal values for computing the expected improvement
   */
  void updateExpectedImprovement();
  virtual void forwardPass(const double stepLength);

  /**
   * @brief Return the threshold used for accepting step along ascent direction
   */
  double get_th_acceptnegstep() const;

  /**
   * @brief Modify the threshold used for accepting step along ascent direction
   */
  void set_th_acceptnegstep(const double th_acceptnegstep);

 protected:
  double dg_;  //!< Internal data for computing the expected improvement
  double dq_;  //!< Internal data for computing the expected improvement
  double dv_;  //!< Internal data for computing the expected improvement

 private:
  double th_acceptnegstep_;  //!< Threshold used for accepting step along ascent direction
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_FDDP_HPP_
