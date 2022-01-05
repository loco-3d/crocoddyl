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

enum EqualitySolverType { LuNull = 0, QrNull, Schur };

class SolverIntro : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the INTRO solver
   *
   * @param[in] problem  Shooting problem
   * @param[in] reduced  Used reduced Schur complement approach (default true)
   */
  explicit SolverIntro(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverIntro();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
                     const bool is_feasible = false, const double regInit = 1e-9);
  virtual double tryStep(const double step_length = 1);
  virtual double stoppingCriteria();
  virtual void resizeData();
  virtual double calcDiff();
  virtual void computeValueFunction(const std::size_t t, const boost::shared_ptr<ActionModelAbstract>& model);
  virtual void computeGains(const std::size_t t);

  /**
   * @brief Return the type of solver used for handling the equality constraints
   *
   */
  EqualitySolverType get_equality_solver() const;

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
   * @brief Modify the type of solver used for handling the equality constraints
   *
   * Note that the default solver is nullspace LU. When we enable parallelization, this strategy is is generally faster
   * than others for medium to large systems.
   */
  void set_equality_solver(const EqualitySolverType type);

  /**
   * @brief Modify the rho parameter used in the merit function
   */
  void set_rho(const double rho);

 protected:
  enum EqualitySolverType eq_solver_;  //!< Strategy used for handling the equality constraints
  double rho_;                         //!< Parameter used in the merit function to predict the expected reduction
  double dPhi_;                        //!< Reduction in the merit function obtained by `tryStep()`
  double dPhiexp_;                     //!< Expected reduction in the merit function
  double hfeas_try_;                   //!< Feasibility of the equality constraint computed by the line search
  double upsilon_;  //!< Estimated penalty paramter that balances relative contribution of the cost function and
                    //!< equality constraints

  std::vector<std::size_t> Hu_rank_;  //!< Rank of the control Jacobian of the equality constraints
  std::vector<Eigen::MatrixXd> KQuu_tmp_;
  std::vector<Eigen::MatrixXd>
      YZ_;  //!< Span \f$\mathbf{Y}\in\mathbb{R}^{rank}\f$ and kernel \f$\mathbf{Z}\in\mathbb{R}^{nullity}\f$ of the
            //!< control-equality constraints \f$\mathbf{H_u}\f$
  std::vector<Eigen::MatrixXd>
      HuY_;  //!< Span-projected Jacobian of the equality-constraint with respect to the control
  std::vector<Eigen::VectorXd> Qz_;     //!< Jacobian of the Hamiltonian \f$\mathbf{Q_{z}}\f$
  std::vector<Eigen::MatrixXd> Qzz_;    //!< Reduced Hessian of the Hamiltonian \f$\mathbf{Q_{zz}}\f$
  std::vector<Eigen::MatrixXd> Qxz_;    //!< Reduced Hessian of the Hamiltonian \f$\mathbf{Q_{xz}}\f$
  std::vector<Eigen::MatrixXd> Quz_;    //!< Reduced Hessian of the Hamiltonian \f$\mathbf{Q_{uz}}\f$
  std::vector<Eigen::VectorXd> k_z_;    //!< Feedforward term in the nullspace of \f$\mathbf{H_u}\f$
  std::vector<Eigen::MatrixXd> K_z_;    //!< Feedback gain in the nullspace of \f$\mathbf{H_u}\f$
  std::vector<Eigen::VectorXd> k_hat_;  //!< Feedforward term related to the equality constraints
  std::vector<Eigen::MatrixXd> K_hat_;  //!< Feedback gain related to the equality constraints
  std::vector<Eigen::MatrixXd> QuuinvHuT_;
  std::vector<Eigen::LLT<Eigen::MatrixXd> > Qzz_llt_;  //!< Cholesky LLT solver
  std::vector<Eigen::FullPivLU<Eigen::MatrixXd> >
      Hu_lu_;  //!< Full-pivot LU solvers used for computing the span and nullspace matrices
  std::vector<Eigen::ColPivHouseholderQR<Eigen::MatrixXd> >
      Hu_qr_;  //!< Column-pivot QR solvers used for computing the span and nullspace matrices
  std::vector<Eigen::PartialPivLU<Eigen::MatrixXd> >
      HuY_lu_;  //!< Partial-pivot LU solvers used for computing the feedforward and feedback gain related to the
                //!< equality constraint
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_INTRO_HPP_