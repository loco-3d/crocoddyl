///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_INTRO_HPP_
#define CROCODDYL_CORE_SOLVERS_INTRO_HPP_

#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {

template <typename _Scalar>
class SolverIntroTpl : public SolverFDDPTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverIntroTpl)

  typedef _Scalar Scalar;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;
  typedef SolverFDDPTpl<Scalar> SolverFDDP;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ShootingProblem::ActionModelAbstract ActionModelAbstract;
  typedef typename ShootingProblem::ActionDataAbstract ActionDataAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixXsRowMajor MatrixXsRowMajor;

  /**
   * @brief Initialize the INTRO solver
   *
   * @param[in] problem      Shooting problem
   * @param[in] eq_solver    Type of equality solver
   * @param[in] term_solver  Type of terminal solver
   */
  explicit SolverIntroTpl(std::shared_ptr<ShootingProblem> problem,
                          const DynamicsSolverType dyn_solver = FeasShoot,
                          const EqualitySolverType eq_solver = LuNull,
                          const EqualitySolverType term_solver = LuNull);
  virtual ~SolverIntroTpl() = default;

  /**
   * @brief Refresh the Intro linearization and constraint factorizations.
   *
   * The method first delegates to `SolverFDDP::calcDir()` to update the
   * first- and second-order derivatives of the shooting problem. It then
   * processes the control-equality constraints according to the selected
   * equality solver (null-space or Schur complement), storing the factors that
   * are reused during the backward pass.
   */
  virtual void calcDir() override;

  /**
   * @copybrief SolverFDDP::computeDirection
   */
  virtual void computeDirection(const bool recalc = true) override;

  /**
   * @copybrief SolverFDDP::computePolicy
   */
  virtual void computePolicy(const std::size_t t) override;

  /**
   * @copybrief SolverFDDP::computeBatchPolicy
   */
  virtual void computeBatchPolicy(const std::size_t t) override;

  /**
   * @copybrief SolverFDDP::computeValueFunction
   */
  virtual void computeValueFunction(
      const std::size_t t,
      const std::shared_ptr<ActionModelAbstract>& model) override;

  /**
   * @copybrief SolverFDDP::computeBatchValueFunction
   */
  virtual void computeBatchValueFunction(const std::size_t t) override;

  /**
   * @brief Cast the Intro solver to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return SolverIntroTpl<NewScalar> An Intro solver with the new scalar type.
   */
  template <typename NewScalar>
  SolverIntroTpl<NewScalar> cast() const;

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
  const std::vector<MatrixXs>& get_YZ() const;

  /**
   * @brief Return Hessian of the reduced Hamiltonian \f$\mathbf{Q_{zz}}\f$
   */
  const std::vector<MatrixXs>& get_Qzz() const;

  /**
   * @brief Return Hessian of the reduced Hamiltonian \f$\mathbf{Q_{xz}}\f$
   */
  const std::vector<MatrixXs>& get_Qxz() const;

  /**
   * @brief Return Hessian of the reduced Hamiltonian \f$\mathbf{Q_{uz}}\f$
   */
  const std::vector<MatrixXs>& get_Quz() const;

  /**
   * @brief Return Jacobian of the reduced Hamiltonian \f$\mathbf{Q_{z}}\f$
   */
  const std::vector<VectorXs>& get_Qz() const;

  /**
   * @brief Return span-projected Jacobian of the equality-constraint with
   * respect to the control
   */
  const std::vector<MatrixXs>& get_Hy() const;

  /**
   * @brief Return feedforward term related to the nullspace of
   * \f$\mathbf{H_u}\f$
   */
  const std::vector<VectorXs>& get_kz() const;

  /**
   * @brief Return feedback gain related to the nullspace of \f$\mathbf{H_u}\f$
   */
  const std::vector<MatrixXs>& get_Kz() const;

  /**
   * @brief Return feedforward term related to the equality constraints
   */
  const std::vector<VectorXs>& get_ks() const;

  /**
   * @brief Return feedback gain related to the equality constraints
   */
  const std::vector<MatrixXs>& get_Ks() const;

  /**
   * @brief Return Hessian of the reduced Hamiltonian \f$\mathbf{Q_{zc}}\f$
   */
  const std::vector<MatrixXs>& get_Qzc() const;

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
  virtual void resizeRunningData() override;

  /**
   * @copybrief SolverAbstract::resizeTerminalData
   */
  virtual void resizeTerminalData() override;

  using SolverAbstract::feasnorm_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::th_gaptol_;
  using SolverAbstract::us_;
  using SolverAbstract::xs_;
  using SolverFDDP::alphas_;
  using SolverFDDP::callbacks_;
  using SolverFDDP::dyn_solver_;
  using SolverFDDP::fs_;
  using SolverFDDP::K_;
  using SolverFDDP::k_;
  using SolverFDDP::Kc_;
  using SolverFDDP::problem_;
  using SolverFDDP::Qu_;
  using SolverFDDP::Quc_;
  using SolverFDDP::Quu_;
  using SolverFDDP::Quu_llt_;
  using SolverFDDP::Quuk_;
  using SolverFDDP::Qx_;
  using SolverFDDP::Qxu_;
  using SolverFDDP::Qxx_;
  using SolverFDDP::reg_decfactor_;
  using SolverFDDP::reg_incfactor_;
  using SolverFDDP::reg_max_;
  using SolverFDDP::reg_min_;
  using SolverFDDP::rho_;
  using SolverFDDP::term_solver_;
  using SolverFDDP::th_acceptminstep_;
  using SolverFDDP::th_acceptnegstep_;
  using SolverFDDP::th_acceptstep_;
  using SolverFDDP::th_grad_;
  using SolverFDDP::th_minfeas_;
  using SolverFDDP::th_minimprove_;
  using SolverFDDP::th_noimprovement_;
  using SolverFDDP::th_stepdec_;
  using SolverFDDP::th_stepinc_;
  using SolverFDDP::th_stop_;
  using SolverFDDP::Ts_;
  using SolverFDDP::upsilon_decfactor_;
  using SolverFDDP::Vx_;
  using SolverFDDP::Vxc_;
  using SolverFDDP::Vxx_;
  using SolverFDDP::Vxx_f_;
  using SolverFDDP::Vxx_tmp_;
  using SolverFDDP::zero_upsilon_;

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
  std::vector<MatrixXsRowMajor> KQuu_2Qxu_;
  std::vector<MatrixXs>
      YZ_;  //!< Span \f$\mathbf{Y}\in\mathbb{R}^{rank}\f$ and kernel
            //!< \f$\mathbf{Z}\in\mathbb{R}^{nullity}\f$ of the control-equality
            //!< constraints \f$\mathbf{H_u}\f$
  std::vector<MatrixXs>
      Hy_;  //!< Span-projected Jacobian of the equality-constraint with respect
            //!< to the control
  std::vector<VectorXs>
      Qz_;  //!< Jacobian of the reduced Hamiltonian \f$\mathbf{Q_{z}}\f$
  std::vector<MatrixXs>
      Qzz_;  //!< Hessian of the reduced Hamiltonian \f$\mathbf{Q_{zz}}\f$
  std::vector<MatrixXs>
      Qxz_;  //!< Hessian of the reduced Hamiltonian \f$\mathbf{Q_{xz}}\f$
  std::vector<MatrixXs>
      Quz_;  //!< Hessian of the reduced Hamiltonian \f$\mathbf{Q_{uz}}\f$
  std::vector<VectorXs>
      kz_;  //!< Feedforward term in the nullspace of \f$\mathbf{H_u}\f$
  std::vector<MatrixXs>
      Kz_;  //!< Feedback gain in the nullspace of \f$\mathbf{H_u}\f$
  std::vector<VectorXs>
      ks_;  //!< Feedforward term related to the equality constraints
  std::vector<MatrixXs>
      Ks_;  //!< Feedback gain related to the equality constraints
  std::vector<MatrixXs> QuuinvHuT_;
  std::vector<Eigen::LLT<MatrixXs> > Qzz_llt_;  //!< Cholesky LLT solver
  std::vector<Eigen::FullPivLU<MatrixXs> >
      Hu_lu_;  //!< Full-pivot LU solvers used for computing the span and
               //!< nullspace matrices
  std::vector<Eigen::ColPivHouseholderQR<MatrixXs> >
      Hu_qr_;  //!< Column-pivot QR solvers used for computing the span and
               //!< nullspace matrices
  std::vector<Eigen::PartialPivLU<MatrixXs> >
      Hy_lu_;  //!< Partial-pivot LU solvers used for computing the feedforward
               //!< and feedback gain related to the equality constraint

  std::vector<MatrixXs> Kcs_;
  std::vector<MatrixXs> QuuKc_Quc_;
  std::vector<MatrixXs>
      Qzc_;  //!< Hessian of the reduced Hamiltonian \f$\mathbf{Q_{zc}}\f$
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/solvers/intro.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SolverIntroTpl)

#endif  // CROCODDYL_CORE_SOLVERS_INTRO_HPP_
