///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_ODYN_SQP_HPP_
#define CROCODDYL_CORE_SOLVERS_ODYN_SQP_HPP_

#include <odyn/data.hpp>
#include <odyn/model.hpp>
#include <odyn/params.hpp>

#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Feasibility-driven Differential Dynamic Programming (OdynSQP) solver
 *
 * The OdynSQP solver computes an optimal trajectory and control commands by
 * iterates running `backwardPass()` and `forwardPass()`. The backward pass
 * accepts infeasible guess as described in the `SolverOdynSQP::backwardPass()`.
 * Additionally, the forward pass handles infeasibility simulations that
 * resembles the numerical behaviour of a multiple-shooting formulation, i.e.:
 * \f{eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0 - (1 -
 * \alpha)\mathbf{\bar{f}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
 * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\ \mathbf{\hat{x}}_{k+1} &=&
 * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k) - (1 -
 * \alpha)\mathbf{\bar{f}}_{k+1}.
 * \f}
 * Note that the forward pass keeps the gaps \f$\mathbf{\bar{f}}_s\f$ open
 * according to the step length \f$\alpha\f$ that has been accepted. This solver
 * has shown empirically greater globalization strategy. Additionally, the
 * expected improvement computation considers the gaps in the dynamics:
 * \f{equation}
 *   \Delta J(\alpha) = \Delta_1\alpha + \frac{1}{2}\Delta_2\alpha^2,
 * \f}
 * with
 * \f{eqnarray}
 *   \Delta_1 = \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k}
 * +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k} -
 *   V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
 *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k +
 * \mathbf{\bar{f}}_k^\top(2 V_{\mathbf{xx}_k}\mathbf{x}_k
 * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). \f}
 *
 * For more details about the feasibility-driven differential dynamic
 * programming algorithm see: \include mastalli-icra20.bib
 *
 * \sa `SolverAbstract()`, `backwardPass()`, `forwardPass()`, and
 * `expectedImprovement()`.
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
   * @brief Check if we should accept or not the step
   *
   * @return True if we should accept the step. False otherwise
   */
  virtual bool checkAcceptance() override;

  /**
   * @brief Update the merit function value for the current guess
   */
  virtual void updateMeritFunction() override;

  /**
   * @copybrief SolverAbstract::computeCandidate
   */
  virtual void computeCandidate(const Scalar step_length = Scalar(1.)) override;

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
  odyn::ModelTpl<Scalar, odyn::SparseBackend> model_;
  odyn::DataTpl<Scalar, odyn::SparseBackend> data_;
  odyn::ParamsTpl<Scalar> params_;
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
  using SolverAbstract::ffeas_;
  using SolverAbstract::ffeas_try_;
  using SolverAbstract::fs_;
  using SolverAbstract::fs_try_;
  using SolverAbstract::gfeas_;
  using SolverAbstract::gfeas_try_;
  using SolverAbstract::hfeas_;
  using SolverAbstract::hfeas_try_;
  using SolverAbstract::iter_;
  using SolverAbstract::merit_;
  using SolverAbstract::nh_T_;
  using SolverAbstract::preg_;
  using SolverAbstract::problem_;
  using SolverAbstract::reg_max_;
  using SolverAbstract::reg_min_;
  using SolverAbstract::steplength_;
  using SolverAbstract::stop_;
  using SolverAbstract::th_acceptstep_;
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
