///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_ODYN_SQP_HPP_
#define CROCODDYL_CORE_SOLVERS_ODYN_SQP_HPP_

#include <odyn/data.hpp>
#include <odyn/model.hpp>
#include <odyn/params.hpp>
#include <odyn/solver.hpp>

#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Odyn-based Sequential Quadratic Programming (SQP) solver
 *
 * This solver wraps Odyn’s sparse QP engine to solve Crocoddyl shooting
 * problems with equality and inequality constraints. At each iteration it
 * builds a sparse QP in `computeQuadraticModel()`, solves it with Odyn, and
 * then maps the QP step back to state/control updates in
 * `computeSearchDirection()` / `tryStep()`. The try step supports
 * infeasible iterates (open dynamics gaps) and line-searches them, which
 * improves globalization on hard-constrained problems. The expected improvement
 * calculation also accounts for the dynamics gaps.
 *
 * See `SolverAbstract()`, `computeSearchDirection()`, `tryStep()`, and
 * `expectedImprovement()` for the main iteration steps.
 */
template <typename _Scalar>
class SolverOdynSQPTpl : public SolverAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverOdynSQPTpl)

  typedef _Scalar Scalar;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ShootingProblem::ActionModelAbstract ActionModelAbstract;
  typedef typename ShootingProblem::ActionDataAbstract ActionDataAbstract;
  typedef CallbackAbstractTpl<Scalar> CallbackAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixXsRowMajor MatrixXsRowMajor;
  using Solver = typename odyn::SparseQPTpl<Scalar>;
  using Model = typename odyn::SparseModelTpl<Scalar>;
  using Data = typename odyn::SparseDataTpl<Scalar>;
  using Params = typename odyn::ParamsTpl<Scalar>;
  using SolverAbstract::computeDynamicFeasibility;
  using SolverAbstract::computeEqualityFeasibility;
  using SolverAbstract::computeFeasibility;
  using SolverAbstract::computeInequalityFeasibility;
  using SolverAbstract::resizeData;

  /**
   * @brief Initialize the OdynSQP solver
   *
   * @param[in] problem      Shooting problem
   * @param[in] dyn_solver   Type of dynamic solver
   * @param[in] term_solver  Type of terminal solver
   */
  explicit SolverOdynSQPTpl(std::shared_ptr<ShootingProblem> problem);
  virtual ~SolverOdynSQPTpl() = default;

  /**
   * @copybrief SolverAbstract::computeDirection
   */
  virtual void computeDirection(const bool recalc = true) override;

  /**
   * @copybrief SolverAbstract::stoppingCriteria
   */
  virtual Scalar stoppingCriteria() override;

  /**
   * @copybrief SolverAbstract::expectedImprovement
   */
  virtual Vector3s expectedImprovement() override;

  /**
   * @copybrief SolverAbstract::computeMeritFunctionImprovement
   */
  virtual void computeMeritFunctionImprovement() override;

  /**
   * @copybrief SolverAbstract::computeExpectedMeritFunctionImprovement
   */
  virtual void computeExpectedMeritFunctionImprovement() override;

  /**
   * @brief Update the merit function value for the current guess
   */
  virtual void updateMeritFunction() override;

  /**
   * @brief Check if we should accept or not the step
   *
   * @return True if we should accept the step. False otherwise
   */
  virtual bool checkAcceptance() override;

  /**
   * @copybrief SolverAbstract::computeCandidate
   */
  virtual void computeCandidate(const Scalar step_length = Scalar(1.)) override;

  /**
   * @brief Build the local Quadratic Program (QP) for the current iterate.
   *
   * @details The QP has the form
   * @f[
   *   \min_{x}\ \tfrac{1}{2}\, x^\top Q\, x \;+\; c^\top x
   *   \quad\text{s.t.}\quad
   *   A\,x = b,\;\; G\,x \le h .
   * @f]
   * Here, @p x is the full decision vector over the trajectory and controls,
   * and
   * @p Q, @p c, @p A, @p b, @p G, @p h are the corresponding (sparse,
   * structured) cost and constraint matrices/vectors induced by the problem’s
   * temporal layout.
   *
   * Decision vector ordering:
   * @f[
   *   x = [\, \Delta x_0,\ \Delta u_0,\ \Delta x_1,\ \Delta
   * u_1,\ \ldots,\ \Delta x_T \,].
   * @f]
   *
   * The sparsity pattern reflects time-coupling (dynamics and local
   * costs/constraints).
   *
   * @tparam Scalar Scalar type (e.g., float, double).
   */
  void computeQuadraticModel();

  /**
   * @brief Update the candidate solution: cost, feasibilities, and merit value
   */
  void updateCandidate() override;

  /**
   * @brief Criteria used to decrease regularization
   */
  bool decreaseRegularizationCriteria() override;

  /**
   * @brief Criteria used to increase regularization
   */
  bool increaseRegularizationCriteria() override;

  /**
   * @brief Increase the state and control regularization values by a
   * `regfactor_` factor
   */
  void increaseRegularization() override;

  /**
   * @brief Decrease the state and control regularization values by a
   * `regfactor_` factor
   */
  void decreaseRegularization() override;

  /**
   * @brief Extract the QP direction into the solver data structures
   *
   * @param[in] x  decision vector from the QP solver
   */
  void extractQpDirection(const VectorXs& x);

  /**
   * @brief Cast the OdynSQP solver to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return SolverOdynSQPTpl<NewScalar> A OdynSQP solver with the new scalar
   * type.
   */
  template <typename NewScalar>
  SolverOdynSQPTpl<NewScalar> cast() const;

  /**
   * @brief Return the QP model used in the OdynSQP solver
   */
  Model& qp_model() noexcept;

  /**
   * @brief Return the QP data used in the OdynSQP solver
   */
  Data& qp_data() noexcept;

  /**
   * @brief Return the number of decision variables in the SQP
   */
  std::size_t get_n() const noexcept;

  /**
   * @brief Return the number of equality constraints in the SQP
   */
  std::size_t get_m() const noexcept;

  /**
   * @brief Return the number of inequality constraints in the SQP
   */
  std::size_t get_p() const noexcept;

  /**
   * @brief Return the QP params used in the OdynSQP solver
   */
  Params& qp_params() noexcept;

  /**
   * @brief Return the verbose level used in the OdynSQP solver
   */
  odyn::VerboseLevel get_verbose_level() const noexcept;

  /**
   * @brief Return the regularization factor used to increase the damping value
   */
  Scalar get_reg_incfactor() const;

  /**
   * @brief Return the regularization factor used to decrease the damping value
   */
  Scalar get_reg_decfactor() const;

  /**
   * @brief Return the tolerance of the expected gradient used for testing the
   * step
   */
  Scalar get_th_grad() const;

  /**
   * @brief Return the step-length threshold used to decrease regularization
   */
  Scalar get_th_stepdec() const;

  /**
   * @brief Return the step-length threshold used to increase regularization
   */
  Scalar get_th_stepinc() const;

  /**
   * @brief Return the minimum improvement threshold used to increase
   * regularization
   */
  Scalar get_th_minimprove() const;

  /**
   * @brief Return the threshold used for accepting step along ascent direction
   */
  Scalar get_th_acceptnegstep() const;

  /**
   * @brief Return the threshold used for accepting minimum steps
   */
  Scalar get_th_acceptminstep() const;

  /**
   * @brief Return the rho parameter used in the merit function
   */
  Scalar get_rho() const;

  /**
   * @brief Return the threshold for switching to feasibility
   */
  Scalar get_th_minfeas() const;

  /**
   * @brief Return the estimated penalty parameter that balances relative
   * contribution of the cost function and equality constraints
   */
  Scalar get_upsilon() const;

  /**
   * @brief Return the upsilon decresing factor used to estimate to balance
   * optimality and feasibility
   */
  Scalar get_upsilon_decfactor() const;

  /**
   * @brief Return the zero-upsilon label
   *
   * True if we set the estimated penalty parameter (upsilon) to zero when solve
   * is called.
   */
  bool get_zero_upsilon() const;

  /**
   * @brief Modify the QP params used in the OdynSQP solver
   */
  void set_qp_params(const Params& params);

  /**
   * @brief Modify the verbose level used in the OdynSQP solver
   */
  void set_verbose_level(const odyn::VerboseLevel verbose_level);

  /**
   * @brief Modify the regularization factor used to increase the damping value
   */
  void set_reg_incfactor(const Scalar reg_factor);

  /**
   * @brief Modify the regularization factor used to decrease the damping value
   */
  void set_reg_decfactor(const Scalar reg_factor);

  /**
   * @brief Modify the tolerance of the expected gradient used for testing the
   * step
   */
  void set_th_grad(const Scalar th_grad);

  /**
   * @brief Modify the threshold used to accept steps that cannot be be improved
   * due to numerical errors the th noimprovement object
   */
  void set_th_noimprovement(const Scalar th_noimprovement);

  /**
   * @brief Modify the step-length threshold used to decrease regularization
   */
  void set_th_stepdec(const Scalar th_step);

  /**
   * @brief Modify the step-length threshold used to increase regularization
   */
  void set_th_stepinc(const Scalar th_step);

  /**
   * @brief Modify the minimum improvement threshold used to increase
   * regularization
   */
  void set_th_minimprove(const Scalar th_step);

  /**
   * @brief Modify the threshold used for accepting step along ascent direction
   */
  void set_th_acceptnegstep(const Scalar th_acceptnegstep);

  /**
   * @brief Modify the threshold used for accepting minimum steps
   */
  void set_th_acceptminstep(const Scalar th_acceptminstep);

  /**
   * @brief Modify the rho parameter used in the merit function
   */
  void set_rho(const Scalar rho);

  /**
   * @brief Modify the threshold for switching to feasibility
   */
  void set_th_minfeas(const Scalar th_minfeas);

  /**
   * @brief Modify the upsilon decresing factor used to estimate to balance
   * optimality and feasibility
   */
  void set_upsilon_decfactor(const Scalar th_step);

  /**
   * @brief Modify the zero-upsilon label
   *
   * @param zero_upsilon  True if we set estimated penalty parameter (upsilon)
   * to zero when solve is called.
   */
  void set_zero_upsilon(const bool zero_upsilon);

 protected:
  /**
   * @brief Allocate all the internal data needed for the solver
   */
  void allocateData();

  /**
   * @copybrief SolverAbstract::resizeRunningData
   */
  virtual void resizeRunningData() override;

  /**
   * @copybrief SolverAbstract::resizeTerminalData
   */
  virtual void resizeTerminalData() override;

  void updateStateAndControlIndex();

  Scalar reg_incfactor_;  //!< Regularization factor used to increase the
                          //!< damping value
  Scalar reg_decfactor_;  //!< Regularization factor used to decrease the
                          //!< damping value
  Scalar th_grad_;  //!< Tolerance of the expected gradient used for testing the
                    //!< step
  Scalar th_noimprovement_;  //!< Threshold used to accept steps that cannot be
                             //!< be improved due to numerical errors
  Scalar
      th_stepdec_;  //!< Step-length threshold used to decrease regularization
  Scalar
      th_stepinc_;  //!< Step-length threshold used to increase regularization
  Scalar th_minimprove_;     //!< Minimum improvement threshold used in the
                             //!< regularization scheme
  Scalar th_acceptnegstep_;  //!< Threshold used for accepting step along ascent
                             //!< direction
  Scalar th_acceptminstep_;  //!< Threshold used for accepting step along with a
                             //!< minimum length
  Scalar rho_;         //!< Parameter used in the merit function to predict the
                       //!< expected reduction
  Scalar th_minfeas_;  //!< Threshold for switching to feasibility
  Scalar
      upsilon_;  //!< Estimated penalty parameter that balances relative
                 //!< contribution of the cost function and equality constraints
  Scalar upsilon_decfactor_;  //!< Estimated penalty parameter factor used to
                              //!< decrease its value
  bool zero_upsilon_;  //!< True if we wish to set estimated penalty parameter
                       //!< (upsilon) to zero when solve is called.

  std::size_t n_;
  std::size_t m_;
  std::size_t p_;
  VectorXs x_;
  Solver qp_solver_;
  Model qp_model_;
  Data qp_data_;
  Params qp_params_;
  odyn::VerboseLevel verbose_level_;
  std::vector<std::size_t> xs_idx_;
  std::vector<std::size_t> us_idx_;
  std::vector<VectorXs>
      Lxx_dx_;  //!< Second-order change of the cost function
                //!< \f$\boldsymbol{\ell}_{\mathbf{{xx}}}\delta\mathbf{x}\f$
  std::vector<VectorXs>
      Luu_du_;  //!< Second-order change of the cost function
                //!< \f$\boldsymbol{\ell}_{\mathbf{{uu}}}\delta\mathbf{u}\f$
  std::vector<VectorXs>
      Lxu_du_;  //!< Second-order change of the cost function
                //!< \f$\boldsymbol{\ell}_{\mathbf{{xu}}}\delta\mathbf{u}\f$

  using SolverAbstract::acceptstep_;
  using SolverAbstract::alphas_;
  using SolverAbstract::callbacks_;
  using SolverAbstract::cost_;
  using SolverAbstract::cost_try_;
  using SolverAbstract::dfeas_;
  using SolverAbstract::dImpr_;
  using SolverAbstract::dPhi_;
  using SolverAbstract::dPhiexp_;
  using SolverAbstract::dreg_;
  using SolverAbstract::dus_;
  using SolverAbstract::DV_;
  using SolverAbstract::dV_;
  using SolverAbstract::dVexp_;
  using SolverAbstract::dVexp_full_;
  using SolverAbstract::dxs_;
  using SolverAbstract::feas_;
  using SolverAbstract::feasnorm_;
  using SolverAbstract::ffeas_;
  using SolverAbstract::ffeas_try_;
  using SolverAbstract::fs_;
  using SolverAbstract::fs_try_;
  using SolverAbstract::gfeas_;
  using SolverAbstract::gfeas_try_;
  using SolverAbstract::hfeas_;
  using SolverAbstract::hfeas_try_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::iter_;
  using SolverAbstract::merit_;
  using SolverAbstract::ng_T_;
  using SolverAbstract::nh_T_;
  using SolverAbstract::preg_;
  using SolverAbstract::problem_;
  using SolverAbstract::reg_max_;
  using SolverAbstract::reg_min_;
  using SolverAbstract::steplength_;
  using SolverAbstract::stop_;
  using SolverAbstract::th_acceptstep_;
  using SolverAbstract::th_gaptol_;
  using SolverAbstract::th_stop_;
  using SolverAbstract::us_;
  using SolverAbstract::us_try_;
  using SolverAbstract::xs_;
  using SolverAbstract::xs_try_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/solvers/odyn-sqp.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SolverOdynSQPTpl)

#endif  // CROCODDYL_CORE_SOLVERS_ODYN_SQP_HPP_
