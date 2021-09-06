///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_INTRO_HPP_
#define CROCODDYL_CORE_SOLVERS_INTRO_HPP_

#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {

class SolverIntro : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the INTRO solver
   *
   * @param[in] problem  Shooting problem
   */
  explicit SolverIntro(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverIntro();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
                     const bool is_feasible = false, const double regInit = 1e-9);
  virtual double tryStep(const double step_length = 1);
  virtual double stoppingCriteria();

  /**
   * @brief Run the backward pass (Riccati sweep)
   *
   * It assumes that the Jacobian and Hessians of the optimal control problem have been compute (i.e. `calcDiff()`).
   * The backward pass handles infeasible guess through a modified Riccati sweep:
   * \f{eqnarray*}
   *   \mathbf{Q}_{\mathbf{x}_k} &=& \mathbf{l}_{\mathbf{x}_k} + \mathbf{f}^\top_{\mathbf{x}_k} (V_{\mathbf{x}_{k+1}}
   * +
   * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
   *   \mathbf{Q}_{\mathbf{u}_k} &=& \mathbf{l}_{\mathbf{u}_k} + \mathbf{f}^\top_{\mathbf{u}_k} (V_{\mathbf{x}_{k+1}}
   * +
   * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
   *   \mathbf{Q}_{\mathbf{xx}_k} &=& \mathbf{l}_{\mathbf{xx}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
   * V_{\mathbf{xx}_{k+1}}
   * \mathbf{f}_{\mathbf{x}_k},\\
   *   \mathbf{Q}_{\mathbf{xu}_k} &=& \mathbf{l}_{\mathbf{xu}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
   * V_{\mathbf{xx}_{k+1}}
   * \mathbf{f}_{\mathbf{u}_k},\\
   *   \mathbf{Q}_{\mathbf{uu}_k} &=& \mathbf{l}_{\mathbf{uu}_k} + \mathbf{f}^\top_{\mathbf{u}_k}
   * V_{\mathbf{xx}_{k+1}} \mathbf{f}_{\mathbf{u}_k}, \f} where
   * \f$\mathbf{l}_{\mathbf{x}_k}\f$,\f$\mathbf{l}_{\mathbf{u}_k}\f$,\f$\mathbf{f}_{\mathbf{x}_k}\f$ and
   * \f$\mathbf{f}_{\mathbf{u}_k}\f$ are the Jacobians of the cost function and dynamics,
   * \f$\mathbf{l}_{\mathbf{xx}_k}\f$,\f$\mathbf{l}_{\mathbf{xu}_k}\f$ and \f$\mathbf{l}_{\mathbf{uu}_k}\f$ are the
   * Hessians of the cost function, \f$V_{\mathbf{x}_{k+1}}\f$ and \f$V_{\mathbf{xx}_{k+1}}\f$ defines the
   * linear-quadratic approximation of the Value function, and \f$\mathbf{\bar{f}}_{k+1}\f$ describes the gaps of the
   * dynamics.
   */
  virtual void backwardPass();

  /**
   * @brief Compute the feedforward and feedback terms using a Cholesky decomposition
   *
   * To compute the feedforward \f$\mathbf{k}_k\f$ and feedback \f$\mathbf{K}_k\f$ terms, we use a Cholesky
   * decomposition to solve \f$\mathbf{Q}_{\mathbf{uu}_k}^{-1}\f$ term:
   * \f{eqnarray}
   * \mathbf{k}_k &=& \mathbf{Q}_{\mathbf{uu}_k}^{-1}\mathbf{Q}_{\mathbf{u}},\\
   * \mathbf{K}_k &=& \mathbf{Q}_{\mathbf{uu}_k}^{-1}\mathbf{Q}_{\mathbf{ux}}.
   * \f}
   *
   * Note that if the Cholesky decomposition fails, then we re-start the backward pass and increase the
   * state and control regularization values.
   */
  virtual void computeGains(const std::size_t t);

  /**
   * @brief Return the rho parameter used in the merit function
   */
  double get_rho() const;

  /**
   * @brief Return the reduction in the merit function
   */
  double get_dPhi() const;

  /**
   * @brief Return the expected reduction in the merit function
   */
  double get_dPhiexp() const;

  /**
   * @brief Return the estimated penalty parameter that balances relative contribution
   * of the cost function and equality constraints
   */
  double get_upsilon() const;

  /**
   * @brief Modify the rho parameter used in the merit function
   */
  void set_rho(const double rho);

 protected:
  double rho_;        //!< Parameter used in the merit function to predict the expected reduction
  double dPhi_;       //!< Reduction in the merit function obtained by `tryStep()`
  double dPhiexp_;    //!< Expected reduction in the merit function
  double hfeas_try_;  //!< Feasibility of the equality constraint computed by the line search
  double upsilon_;    //!< Estimated penalty paramter that balances relative contribution of the cost function and
                      //!< equality constraints

  Eigen::MatrixXd QuuK_tmp_;
  std::vector<Eigen::VectorXd> k_hat_;
  std::vector<Eigen::MatrixXd> K_hat_;
  std::vector<Eigen::MatrixXd> Quu_hat_;
  std::vector<Eigen::MatrixXd> QuuinvHuT_;
  std::vector<Eigen::LLT<Eigen::MatrixXd> > Quu_hat_llt_;  //!< Cholesky LLT solver
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_INTRO_HPP_