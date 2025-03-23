///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_INTRO_HPP_
#define CROCODDYL_CORE_SOLVERS_INTRO_HPP_

#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {

enum EqualitySolverType { LuNull = 0, QrNull, Schur };

class SolverIntro : public SolverFDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the INTRO solver
   *
   * @param[in] problem  Shooting problem
   * @param[in] reduced  Use the reduced Schur-complement approach (default
   * true)
   */
  explicit SolverIntro(std::shared_ptr<ShootingProblem> problem);
  virtual ~SolverIntro();

  virtual bool solve(
      const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
      const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const double init_reg = NAN);
  virtual double tryStep(const double step_length = 1);
  virtual double stoppingCriteria();
  virtual void resizeData();
  virtual double calcDiff();
  virtual void computeValueFunction(
      const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model);
  virtual void computeGains(const std::size_t t);

  /**
   * @brief Return the type of solver used for handling the equality constraints
   */
  EqualitySolverType get_equality_solver() const;

  /**
   * @brief Return the threshold for switching to feasibility
   */
  double get_th_feas() const;

  /**
   * @brief Return the rho parameter used in the merit function
   */
  double get_rho() const;

  /**
   * @brief Return the estimated penalty parameter that balances relative
   * contribution of the cost function and equality constraints
   */
  double get_upsilon() const;

  /**
   * @brief Return the rank of control-equality constraints \f$\mathbf{H_u}\f
   */
  const std::vector<std::size_t>& get_Hu_rank() const;

  /**
   * @brief Return the span and kernel of control-equality constraints
   * \f$\mathbf{H_u}\f
   */
  const std::vector<Eigen::MatrixXd>& get_YZ() const;

  /**
   * @brief Return Hessian of the reduced Hamiltonian \f$\mathbf{Q_{zz}}\f$
   */
  const std::vector<Eigen::MatrixXd>& get_Qzz() const;

  /**
   * @brief Return Hessian of the reduced Hamiltonian \f$\mathbf{Q_{xz}}\f$
   */
  const std::vector<Eigen::MatrixXd>& get_Qxz() const;

  /**
   * @brief Return Hessian of the reduced Hamiltonian \f$\mathbf{Q_{uz}}\f$
   */
  const std::vector<Eigen::MatrixXd>& get_Quz() const;

  /**
   * @brief Return Jacobian of the reduced Hamiltonian \f$\mathbf{Q_{z}}\f$
   */
  const std::vector<Eigen::VectorXd>& get_Qz() const;

  /**
   * @brief Return span-projected Jacobian of the equality-constraint with
   * respect to the control
   */
  const std::vector<Eigen::MatrixXd>& get_Hy() const;

  /**
   * @brief Return feedforward term related to the nullspace of
   * \f$\mathbf{H_u}\f$
   */
  const std::vector<Eigen::VectorXd>& get_kz() const;

  /**
   * @brief Return feedback gain related to the nullspace of \f$\mathbf{H_u}\f$
   */
  const std::vector<Eigen::MatrixXd>& get_Kz() const;

  /**
   * @brief Return feedforward term related to the equality constraints
   */
  const std::vector<Eigen::VectorXd>& get_ks() const;

  /**
   * @brief Return feedback gain related to the equality constraints
   */
  const std::vector<Eigen::MatrixXd>& get_Ks() const;

  /**
   * @brief Return the zero-upsilon label
   *
   * True if we set the estimated penalty parameter (upsilon) to zero when solve
   * is called.
   */
  bool get_zero_upsilon() const;

  /**
   * @brief Modify the type of solver used for handling the equality constraints
   *
   * Note that the default solver is nullspace LU. When we enable
   * parallelization, this strategy is generally faster than others for medium
   * to large systems.
   */
  void set_equality_solver(const EqualitySolverType type);

  /**
   * @brief Modify the threshold for switching to feasibility
   */
  void set_th_feas(const double th_feas);

  /**
   * @brief Modify the rho parameter used in the merit function
   */
  void set_rho(const double rho);

  /**
   * @brief Modify the zero-upsilon label
   *
   * @param zero_upsilon  True if we set estimated penalty parameter (upsilon)
   * to zero when solve is called.
   */
  void set_zero_upsilon(const bool zero_upsilon);

 protected:
  enum EqualitySolverType
      eq_solver_;   //!< Strategy used for handling the equality constraints
  double th_feas_;  //!< Threshold for switching to feasibility
  double rho_;      //!< Parameter used in the merit function to predict the
                    //!< expected reduction
  double
      upsilon_;  //!< Estimated penalty parameter that balances relative
                 //!< contribution of the cost function and equality constraints
  bool zero_upsilon_;  //!< True if we wish to set estimated penalty parameter
                       //!< (upsilon) to zero when solve is called.

  std::vector<std::size_t>
      Hu_rank_;  //!< Rank of the control Jacobian of the equality constraints
  std::vector<Eigen::MatrixXd> KQuu_tmp_;
  std::vector<Eigen::MatrixXd>
      YZ_;  //!< Span \f$\mathbf{Y}\in\mathbb{R}^{rank}\f$ and kernel
            //!< \f$\mathbf{Z}\in\mathbb{R}^{nullity}\f$ of the control-equality
            //!< constraints \f$\mathbf{H_u}\f$
  std::vector<Eigen::MatrixXd>
      Hy_;  //!< Span-projected Jacobian of the equality-constraint with respect
            //!< to the control
  std::vector<Eigen::VectorXd>
      Qz_;  //!< Jacobian of the reduced Hamiltonian \f$\mathbf{Q_{z}}\f$
  std::vector<Eigen::MatrixXd>
      Qzz_;  //!< Hessian of the reduced Hamiltonian \f$\mathbf{Q_{zz}}\f$
  std::vector<Eigen::MatrixXd>
      Qxz_;  //!< Hessian of the reduced Hamiltonian \f$\mathbf{Q_{xz}}\f$
  std::vector<Eigen::MatrixXd>
      Quz_;  //!< Hessian of the reduced Hamiltonian \f$\mathbf{Q_{uz}}\f$
  std::vector<Eigen::VectorXd>
      kz_;  //!< Feedforward term in the nullspace of \f$\mathbf{H_u}\f$
  std::vector<Eigen::MatrixXd>
      Kz_;  //!< Feedback gain in the nullspace of \f$\mathbf{H_u}\f$
  std::vector<Eigen::VectorXd>
      ks_;  //!< Feedforward term related to the equality constraints
  std::vector<Eigen::MatrixXd>
      Ks_;  //!< Feedback gain related to the equality constraints
  std::vector<Eigen::MatrixXd> QuuinvHuT_;
  std::vector<Eigen::LLT<Eigen::MatrixXd> > Qzz_llt_;  //!< Cholesky LLT solver
  std::vector<Eigen::FullPivLU<Eigen::MatrixXd> >
      Hu_lu_;  //!< Full-pivot LU solvers used for computing the span and
               //!< nullspace matrices
  std::vector<Eigen::ColPivHouseholderQR<Eigen::MatrixXd> >
      Hu_qr_;  //!< Column-pivot QR solvers used for computing the span and
               //!< nullspace matrices
  std::vector<Eigen::PartialPivLU<Eigen::MatrixXd> >
      Hy_lu_;  //!< Partial-pivot LU solvers used for computing the feedforward
               //!< and feedback gain related to the equality constraint
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_INTRO_HPP_
