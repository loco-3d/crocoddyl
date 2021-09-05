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

  std::vector<Eigen::VectorXd> k_hat_;
  std::vector<Eigen::MatrixXd> K_hat_;
  std::vector<Eigen::MatrixXd> Quu_hat_;
  std::vector<Eigen::MatrixXd> QuuinvHuT_;
  std::vector<Eigen::LLT<Eigen::MatrixXd> > Quu_hat_llt_;  //!< Cholesky LLT solver
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_INTRO_HPP_