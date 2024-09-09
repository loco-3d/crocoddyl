///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2024, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_INTRO_HPP_
#define CROCODDYL_CORE_SOLVERS_INTRO_HPP_

#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {

class SolverIntro : public SolverFDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef typename MathBaseTpl<double>::MatrixXsRowMajor MatrixXdRowMajor;

  /**
   * @brief Initialize the INTRO solver
   *
   * @param[in] problem      Shooting problem
   * @param[in] eq_solver    Type of equality solver
   * @param[in] term_solver  Type of terminal solver
   */
  explicit SolverIntro(std::shared_ptr<ShootingProblem> problem,
                       const DynamicsSolverType dyn_solver = FeasShoot,
                       const EqualitySolverType eq_solver = LuNull,
                       const EqualitySolverType term_solver = LuNull);
  virtual ~SolverIntro();

  /**
   * @copybrief SolverFDDP::calcDir
   */
  virtual void calcDir();

  /**
   * @copybrief SolverFDDP::computePolicy
   */
  virtual void computePolicy(const std::size_t t);

  /**
   * @copybrief SolverFDDP::computeBatchPolicy
   */
  virtual void computeBatchPolicy(const std::size_t t);

  /**
   * @copybrief SolverFDDP::computeValueFunction
   */
  virtual void computeValueFunction(
      const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model);

  /**
   * @copybrief SolverFDDP::computeBatchValueFunction
   */
  virtual void computeBatchValueFunction(const std::size_t t);

  /**
   * @brief Return the type of solver used for handling the equality constraints
   */
  EqualitySolverType get_equality_solver() const;

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
   * @brief Return Hessian of the reduced Hamiltonian \f$\mathbf{Q_{zc}}\f$
   */
  const std::vector<Eigen::MatrixXd>& get_Qzc() const;

  /**
   * @brief Modify the type of solver used for handling the equality constraints
   *
   * Note that the default solver is nullspace LU. When we enable
   * parallelization, this strategy is generally faster than others for medium
   * to large systems.
   */
  void set_equality_solver(const EqualitySolverType type);

 protected:
  void allocateData();

  /**
   * @copybrief SolverAbstract::resizeRunningData
   */
  virtual void resizeRunningData();

  /**
   * @copybrief SolverAbstract::resizeTerminalData
   */
  virtual void resizeTerminalData();

  void calcLuNullDir();
  void calcQrNullDir();
  void computeNullPolicy(const std::size_t t);
  void computeNullBatchPolicy(const std::size_t t);
  void computeSchurPolicy(const std::size_t t);
  void computeSchurBatchPolicy(const std::size_t t);

  enum EqualitySolverType
      eq_solver_;  //!< Strategy used for handling the equality constraints

  std::vector<std::size_t>
      Hu_rank_;  //!< Rank of the control Jacobian of the equality constraints
  std::vector<MatrixXdRowMajor> KQuu_2Qxu_;
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

  std::vector<Eigen::MatrixXd> Kcs_;
  std::vector<Eigen::MatrixXd> QuuKc_Quc_;
  std::vector<Eigen::MatrixXd>
      Qzc_;  //!< Hessian of the reduced Hamiltonian \f$\mathbf{Q_{zc}}\f$
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_INTRO_HPP_
