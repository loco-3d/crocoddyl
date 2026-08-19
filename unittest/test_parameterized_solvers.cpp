///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <cmath>
#include <type_traits>

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/integrator/discretized.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/residuals/parameters.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/intro.hpp"
#include "crocoddyl/multibody/dynamics/impulse-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

template <typename Scalar>
struct ParameterizedSolverFixture {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::ActionModelAbstractTpl<Scalar> ActionModel;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> LQR;
  typedef crocoddyl::ConstraintModelAbstractTpl<Scalar> ConstraintModel;
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> ConstraintManager;
  typedef crocoddyl::ConstraintModelResidualTpl<Scalar> ResidualConstraint;
  typedef crocoddyl::LQRParamsTpl<Scalar> LQRParams;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ParameterPhaseModelTpl<Scalar> ParameterPhaseModel;
  typedef crocoddyl::ShootingProblemTpl<Scalar> Problem;
  typedef crocoddyl::ResidualModelParametersTpl<Scalar> ParameterResidual;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  ParameterizedSolverFixture() {
    const std::size_t nx = 4;
    phase0_model = std::make_shared<LQR>(nx, 2, 1, 0, 1, true);
    phase1_model = std::make_shared<LQR>(nx, 1, 2, 0, 1, true);
    terminal_model = std::make_shared<LQR>(nx, 0, 2, 0, 0, true);
    MatrixXs H0 = MatrixXs::Zero(1, nx + 2 + 1);
    H0(0, nx) = Scalar(1.);
    H0(0, nx + 1) = Scalar(0.5);
    H0(0, nx + 2) = Scalar(0.4);
    phase0_model->set_H(H0);
    MatrixXs H1 = MatrixXs::Zero(1, nx + 1 + 2);
    H1(0, nx) = Scalar(1.);
    H1(0, nx + 1) = Scalar(-0.25);
    H1(0, nx + 2) = Scalar(0.3);
    phase1_model->set_H(H1);
    params0 = std::make_shared<ParameterManager>(phase0_model->get_state());
    params1 = std::make_shared<ParameterManager>(phase1_model->get_state());
    params0->addParam(
        "lqr", std::make_shared<LQRParams>(phase0_model->get_state(), 1));
    params1->addParam(
        "lqr", std::make_shared<LQRParams>(phase1_model->get_state(), 2));
    params0->addParam("inactive",
                      std::make_shared<LQRParams>(phase0_model->get_state(), 2),
                      false);
    params1->addParam("inactive",
                      std::make_shared<LQRParams>(phase1_model->get_state(), 1),
                      false);

    const VectorXs reference0 = VectorXs::Constant(1, Scalar(0.15));
    VectorXs reference1(2);
    reference1 << Scalar(-0.2), Scalar(0.3);
    constraints0 =
        std::make_shared<ConstraintManager>(phase0_model->get_state(), 2, 1);
    constraints1 =
        std::make_shared<ConstraintManager>(phase1_model->get_state(), 1, 2);
    constraints0->addConstraint(
        "parameters", std::make_shared<ResidualConstraint>(
                          phase0_model->get_state(),
                          std::make_shared<ParameterResidual>(
                              phase0_model->get_state(), reference0, 2)));
    constraints1->addConstraint(
        "parameters", std::make_shared<ResidualConstraint>(
                          phase1_model->get_state(),
                          std::make_shared<ParameterResidual>(
                              phase1_model->get_state(), reference1, 1)));
    phase_params0 =
        std::make_shared<ParameterPhaseModel>(params0, constraints0);
    phase_params1 =
        std::make_shared<ParameterPhaseModel>(params1, constraints1);

    std::vector<std::vector<std::shared_ptr<ActionModel>>> phases(2);
    phases[0] = {phase0_model, phase0_model};
    phases[1] = {phase1_model, phase1_model};
    problem = std::make_shared<Problem>(
        VectorXs::Zero(nx), phases, terminal_model,
        std::vector<std::shared_ptr<ParameterPhaseModel>>{phase_params0,
                                                          phase_params1});
    problem->set_nthreads(1);
    us = {VectorXs::Constant(2, Scalar(0.1)),
          VectorXs::Constant(2, Scalar(-0.05)),
          VectorXs::Constant(1, Scalar(0.2)),
          VectorXs::Constant(1, Scalar(-0.1))};
    xs.resize(us.size() + 1);
    problem->rollout(us, xs);
  }

  std::shared_ptr<LQR> phase0_model;
  std::shared_ptr<LQR> phase1_model;
  std::shared_ptr<LQR> terminal_model;
  std::shared_ptr<ParameterManager> params0;
  std::shared_ptr<ParameterManager> params1;
  std::shared_ptr<ParameterPhaseModel> phase_params0;
  std::shared_ptr<ParameterPhaseModel> phase_params1;
  std::shared_ptr<ConstraintManager> constraints0;
  std::shared_ptr<ConstraintManager> constraints1;
  std::shared_ptr<Problem> problem;
  std::vector<VectorXs> xs;
  std::vector<VectorXs> us;
};

template <typename Scalar>
struct QuadraticParameterFixture {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::ActionModelAbstractTpl<Scalar> ActionModel;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> LQR;
  typedef crocoddyl::LQRParamsTpl<Scalar> LQRParams;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ParameterPhaseModelTpl<Scalar> ParameterPhaseModel;
  typedef crocoddyl::ShootingProblemTpl<Scalar> Problem;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  QuadraticParameterFixture()
      : initial(VectorXs::Constant(1, Scalar(-0.4))),
        optimum(VectorXs::Constant(1, Scalar(0.35))) {
    running = std::make_shared<LQR>(1, 1, 1, 0, 0, true);
    running->set_A(MatrixXs::Identity(1, 1));
    running->set_B(MatrixXs::Zero(1, 1));
    running->set_P(MatrixXs::Zero(1, 1));
    running->set_Q(MatrixXs::Zero(1, 1));
    running->set_R(MatrixXs::Identity(1, 1));
    running->set_N(MatrixXs::Zero(1, 1));
    running->set_W(MatrixXs::Zero(1, 1));
    running->set_Y(MatrixXs::Zero(1, 1));
    running->set_V(MatrixXs::Zero(1, 1));
    running->set_f(VectorXs::Zero(1));
    running->set_q(VectorXs::Zero(1));
    running->set_r(VectorXs::Zero(1));
    running->set_m(VectorXs::Zero(1));

    terminal = std::make_shared<LQR>(1, 0, 1, 0, 0, true);
    terminal->set_A(MatrixXs::Identity(1, 1));
    terminal->set_P(MatrixXs::Zero(1, 1));
    terminal->set_Q(MatrixXs::Zero(1, 1));
    terminal->set_W(Scalar(2.) * MatrixXs::Identity(1, 1));
    terminal->set_Y(MatrixXs::Zero(1, 1));
    terminal->set_f(VectorXs::Zero(1));
    terminal->set_q(VectorXs::Zero(1));
    terminal->set_m(VectorXs::Constant(1, Scalar(-0.7)));

    params = std::make_shared<ParameterManager>(running->get_state());
    params->addParam("lqr",
                     std::make_shared<LQRParams>(running->get_state(), 1));
    phase_params = std::make_shared<ParameterPhaseModel>(params);
    problem = std::make_shared<Problem>(
        VectorXs::Zero(1),
        std::vector<std::shared_ptr<ActionModel>>(2, running), terminal,
        phase_params);
    problem->set_nthreads(1);
    us.assign(2, VectorXs::Zero(1));
    xs.resize(us.size() + 1);
    problem->rollout(us, xs);
  }

  std::shared_ptr<LQR> running;
  std::shared_ptr<LQR> terminal;
  std::shared_ptr<ParameterManager> params;
  std::shared_ptr<ParameterPhaseModel> phase_params;
  std::shared_ptr<Problem> problem;
  std::vector<VectorXs> xs;
  std::vector<VectorXs> us;
  VectorXs initial;
  VectorXs optimum;
};

template <typename Solver>
struct CandidateExceptionSolver : public Solver {
  typedef typename Solver::Scalar Scalar;

  explicit CandidateExceptionSolver(
      const std::shared_ptr<typename Solver::ProblemAbstract>& problem)
      : Solver(problem) {}

  virtual void computeDirection(const bool recalc = true) override {
    Solver::computeDirection(recalc);
    if (mark_accepted_) {
      this->acceptstep_ = true;
      mark_accepted_ = false;
    }
  }

  virtual void computeCandidate(const Scalar steplength) override {
    Solver::computeCandidate(steplength);
    if (throw_candidates_) {
      throw_pretty("candidate failure");
    }
  }

  void stopThrowing() { throw_candidates_ = false; }
  bool accepted() const { return this->acceptstep_; }

 private:
  bool mark_accepted_ = true;
  bool throw_candidates_ = true;
};

template <typename Scalar, typename Solver>
void check_quadratic_parameter_solver() {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(2e-4) : Scalar(1e-9);
  QuadraticParameterFixture<Scalar> fixture;
  const std::vector<VectorXs> init_p(1, fixture.initial);
  const std::shared_ptr<typename Solver::ProblemAbstract> problem =
      fixture.problem;

  Solver trial(problem);
  BOOST_CHECK(!trial.solve(fixture.xs, fixture.us, init_p, 0, true));
  trial.computeDirection(true);
  trial.expectedImprovement();
  trial.tryStep(Scalar(0.));
  BOOST_CHECK(!trial.checkAcceptance());
  trial.tryStep(Scalar(0.5));
  BOOST_CHECK(trial.checkAcceptance());
  BOOST_CHECK_CLOSE_FRACTION(trial.get_dV(), trial.get_dVexp(), tol);

  Solver solver(problem);
  BOOST_CHECK(solver.solve(fixture.xs, fixture.us, init_p, 10, true));
  BOOST_CHECK_GT(solver.get_iter(), 0);
  BOOST_REQUIRE_EQUAL(solver.get_p().size(), 1);
  BOOST_CHECK(solver.get_p()[0].isApprox(fixture.optimum, tol));
  BOOST_CHECK(fixture.problem->get_paramsData()[0]->params->params->p.isApprox(
      fixture.optimum, tol));
  const Scalar recomputed_cost =
      fixture.problem->calc(solver.get_xs(), solver.get_us());
  BOOST_CHECK_CLOSE_FRACTION(recomputed_cost, solver.get_cost(), tol);
  fixture.problem->calcDiff(solver.get_xs(), solver.get_us());
  BOOST_CHECK(fixture.problem->get_terminalData()->Lp.isZero(tol));
  BOOST_CHECK_CLOSE_FRACTION(recomputed_cost, Scalar(-0.1225), tol);
}

template <typename Scalar>
void check_quadratic_parameter_fddp() {
  check_quadratic_parameter_solver<Scalar, crocoddyl::SolverFDDPTpl<Scalar>>();
}

template <typename Scalar>
void check_quadratic_parameter_intro() {
  check_quadratic_parameter_solver<Scalar, crocoddyl::SolverIntroTpl<Scalar>>();
}

template <typename Scalar>
void check_parameterized_fddp() {
  typedef crocoddyl::SolverFDDPTpl<Scalar> Solver;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename std::conditional<std::is_same<Scalar, double>::value, float,
                                    double>::type OtherScalar;
  ParameterizedSolverFixture<Scalar> fixture;
  Solver solver(fixture.problem, crocoddyl::FeasShoot, crocoddyl::LuNull,
                crocoddyl::AStateNone);
  std::vector<VectorXs> init_p(2);
  init_p[0] = VectorXs::Constant(1, Scalar(-0.4));
  init_p[1] = VectorXs::Constant(2, Scalar(0.5));
  BOOST_CHECK(!solver.solve(fixture.xs, fixture.us, init_p, 0, true));

  BOOST_REQUIRE_EQUAL(solver.get_p().size(), 2);
  BOOST_CHECK_EQUAL(fixture.params0->get_np(), 1);
  BOOST_CHECK_EQUAL(fixture.params1->get_np(), 2);
  BOOST_CHECK_EQUAL(fixture.params0->get_active_set().count("lqr"), 1);
  BOOST_CHECK_EQUAL(fixture.params0->get_inactive_set().count("inactive"), 1);
  BOOST_CHECK_EQUAL(fixture.params1->get_active_set().count("lqr"), 1);
  BOOST_CHECK_EQUAL(fixture.params1->get_inactive_set().count("inactive"), 1);
  BOOST_CHECK(solver.get_p()[0].isApprox(init_p[0]));
  BOOST_CHECK(solver.get_p()[1].isApprox(init_p[1]));
  BOOST_CHECK(solver.get_p_try()[0].isApprox(solver.get_p()[0]));
  BOOST_CHECK(solver.get_p_try()[1].isApprox(solver.get_p()[1]));
  BOOST_CHECK(fixture.problem->get_paramsData()[0]->params->params->p.isApprox(
      solver.get_p()[0]));
  BOOST_CHECK(fixture.problem->get_paramsData()[1]->params->params->p.isApprox(
      solver.get_p()[1]));

  solver.computeDirection(true);
  const bool malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      solver.computeDirection(true);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  BOOST_REQUIRE_EQUAL(solver.get_Vp().size(), fixture.problem->get_T() + 1);
  BOOST_CHECK_EQUAL(solver.get_Vp()[0].size(), 1);
  BOOST_CHECK_EQUAL(solver.get_Vp()[1].size(), 1);
  BOOST_CHECK_EQUAL(solver.get_Vp()[2].size(), 2);
  BOOST_CHECK_EQUAL(solver.get_Vp().back().size(), 2);
  BOOST_CHECK_EQUAL(solver.get_Qp()[0].size(), 1);
  BOOST_CHECK_EQUAL(solver.get_Qp()[2].size(), 2);
  BOOST_CHECK_EQUAL(solver.get_Qpp()[0].rows(), 1);
  BOOST_CHECK_EQUAL(solver.get_Qpp()[2].rows(), 2);
  BOOST_CHECK_EQUAL(solver.get_Qpx()[0].cols(), 4);
  BOOST_CHECK_EQUAL(solver.get_Qpu()[0].cols(), 2);
  BOOST_CHECK_EQUAL(solver.get_Qpu()[2].cols(), 1);
  BOOST_CHECK_EQUAL(solver.get_P()[0].cols(), 1);
  BOOST_CHECK_EQUAL(solver.get_P()[2].cols(), 2);
  for (std::size_t i = 0; i < solver.get_dp().size(); ++i) {
    BOOST_CHECK(solver.get_dp()[i].allFinite());
    BOOST_CHECK(solver.get_Vp_phase()[i].allFinite());
    BOOST_CHECK(solver.get_Vpp_phase()[i].allFinite());
    BOOST_CHECK(solver.get_Vpx_phase()[i].allFinite());
  }

  const Scalar steplength = Scalar(0.25);
  std::vector<VectorXs> expected_p_try(solver.get_p().size());
  for (std::size_t i = 0; i < solver.get_p().size(); ++i) {
    expected_p_try[i] = solver.get_p()[i] + steplength * solver.get_dp()[i];
  }
  solver.computeCandidate(steplength);
  for (std::size_t i = 0; i < solver.get_p().size(); ++i) {
    BOOST_CHECK(solver.get_p_try()[i].allFinite());
    BOOST_CHECK(solver.get_p_try()[i].isApprox(expected_p_try[i]));
  }

  const std::shared_ptr<typename Solver::ProblemAbstract> problem_before =
      solver.get_problem();
  const std::vector<VectorXs> p_before = solver.get_p();
  std::vector<VectorXs> manager_p_before(solver.get_p().size());
  for (std::size_t i = 0; i < manager_p_before.size(); ++i) {
    manager_p_before[i] =
        fixture.problem->get_paramsData()[i]->params->params->p;
  }
  BOOST_CHECK_THROW(solver.template cast<OtherScalar>(), crocoddyl::Exception);
  BOOST_CHECK(solver.get_problem() == problem_before);
  BOOST_CHECK(fixture.problem == problem_before);
  for (std::size_t i = 0; i < p_before.size(); ++i) {
    BOOST_CHECK(solver.get_p()[i].isApprox(p_before[i]));
    BOOST_CHECK(
        fixture.problem->get_paramsData()[i]->params->params->p.isApprox(
            manager_p_before[i]));
  }
  std::vector<VectorXs> wrong_count(1, VectorXs::Zero(1));
  BOOST_CHECK_THROW(solver.solve(fixture.xs, fixture.us, wrong_count, 0, true),
                    crocoddyl::Exception);
  std::vector<VectorXs> wrong_dimension = init_p;
  wrong_dimension[1].resize(1);
  BOOST_CHECK_THROW(
      solver.solve(fixture.xs, fixture.us, wrong_dimension, 0, true),
      crocoddyl::Exception);
}

template <typename Scalar>
void check_rejected_parameter_restoration() {
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> ConstraintManager;
  typedef crocoddyl::ConstraintModelResidualTpl<Scalar> ResidualConstraint;
  typedef crocoddyl::ShootingProblemTpl<Scalar> Problem;
  typedef crocoddyl::ResidualModelParametersTpl<Scalar> ParameterResidual;
  typedef crocoddyl::ParameterPhaseModelTpl<Scalar> ParameterPhaseModel;
  typedef
      typename ParameterPhaseModel::ConstraintDataManager ConstraintDataManager;
  typedef crocoddyl::SolverFDDPTpl<Scalar> Solver;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(2e-5) : Scalar(1e-12);
  QuadraticParameterFixture<Scalar> fixture;
  const std::shared_ptr<ConstraintManager> constraints =
      std::make_shared<ConstraintManager>(fixture.running->get_state(), 1, 1);
  const std::shared_ptr<ParameterResidual> residual =
      std::make_shared<ParameterResidual>(fixture.running->get_state(),
                                          VectorXs::Zero(1), 1);
  constraints->addConstraint("inequality",
                             std::make_shared<ResidualConstraint>(
                                 fixture.running->get_state(), residual,
                                 VectorXs::Constant(1, Scalar(-0.1)),
                                 VectorXs::Constant(1, Scalar(0.1))));
  constraints->addConstraint("inactive",
                             std::make_shared<ResidualConstraint>(
                                 fixture.running->get_state(), residual,
                                 VectorXs::Constant(1, Scalar(-1.)),
                                 VectorXs::Constant(1, Scalar(1.))),
                             false);
  const std::shared_ptr<ParameterPhaseModel> phase_params =
      std::make_shared<ParameterPhaseModel>(fixture.params, constraints);
  const std::shared_ptr<Problem> problem = std::make_shared<Problem>(
      VectorXs::Zero(1),
      std::vector<std::shared_ptr<typename Problem::ActionModelAbstract>>(
          2, fixture.running),
      fixture.terminal, phase_params);
  problem->set_nthreads(1);
  std::vector<VectorXs> xs(fixture.us.size() + 1);
  problem->rollout(fixture.us, xs);
  const VectorXs accepted = VectorXs::Constant(1, Scalar(0.2));
  Solver solver(
      std::static_pointer_cast<typename Solver::ProblemAbstract>(problem));
  solver.solve(xs, fixture.us, std::vector<VectorXs>(1, accepted), 0, true);
  solver.computeDirection(true);
  const std::shared_ptr<ConstraintDataManager>& constraints_data =
      problem->get_paramsData()[0]->constraints;
  BOOST_REQUIRE_EQUAL(constraints->get_inactive_set().count("inactive"), 1);
  BOOST_CHECK(constraints_data->g.isApprox(accepted, tol));

  solver.computeCandidate(Scalar(1.));
  BOOST_CHECK(!solver.get_p_try()[0].isApprox(accepted, tol));
  BOOST_CHECK(constraints_data->g.isApprox(solver.get_p_try()[0], tol));
  solver.calcDir();
  BOOST_CHECK(
      problem->get_paramsData()[0]->params->params->p.isApprox(accepted, tol));
  BOOST_CHECK(constraints_data->g.isApprox(accepted, tol));
  BOOST_CHECK_SMALL(static_cast<double>(solver.get_gfeas() - Scalar(0.1)),
                    static_cast<double>(tol));
  BOOST_CHECK_SMALL(static_cast<double>(solver.get_hfeas()),
                    static_cast<double>(tol));
}

template <typename Scalar>
void check_mixed_parameter_constraint_restoration() {
  typedef crocoddyl::SolverFDDPTpl<Scalar> Solver;
  typedef typename ParameterizedSolverFixture<Scalar>::ActionModel ActionModel;
  typedef typename ParameterizedSolverFixture<Scalar>::ParameterPhaseModel
      ParameterPhaseModel;
  typedef typename ParameterizedSolverFixture<Scalar>::Problem Problem;
  typedef typename ParameterizedSolverFixture<Scalar>::VectorXs VectorXs;
  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(2e-5) : Scalar(1e-12);
  ParameterizedSolverFixture<Scalar> fixture;
  std::vector<std::vector<std::shared_ptr<ActionModel>>> phases(2);
  phases[0] = {fixture.phase0_model, fixture.phase0_model};
  phases[1] = {fixture.phase1_model, fixture.phase1_model};
  const std::shared_ptr<Problem> problem = std::make_shared<Problem>(
      VectorXs::Zero(4), phases, fixture.terminal_model,
      std::vector<std::shared_ptr<ParameterPhaseModel>>{
          std::make_shared<ParameterPhaseModel>(fixture.params0),
          std::make_shared<ParameterPhaseModel>(fixture.params1,
                                                fixture.constraints1)});
  problem->set_nthreads(1);
  std::vector<VectorXs> xs(fixture.us.size() + 1);
  problem->rollout(fixture.us, xs);
  std::vector<VectorXs> initial(2);
  initial[0] = VectorXs::Constant(1, Scalar(0.4));
  initial[1].resize(2);
  initial[1] << Scalar(-0.5), Scalar(0.6);
  Solver solver(
      std::static_pointer_cast<typename Solver::ProblemAbstract>(problem));
  solver.solve(xs, fixture.us, initial, 0, true);
  BOOST_CHECK_NO_THROW(solver.computeDirection(true));

  const std::vector<std::shared_ptr<ParameterPhaseModel>>& params_models =
      problem->get_paramsModel();
  BOOST_REQUIRE_EQUAL(params_models.size(), 2);
  BOOST_CHECK(params_models[0]->get_constraints() == nullptr);
  BOOST_CHECK(problem->get_paramsData()[0]->constraints == nullptr);
  BOOST_REQUIRE(params_models[1]->get_constraints() != nullptr);
  BOOST_REQUIRE(problem->get_paramsData()[1]->constraints != nullptr);

  const std::vector<VectorXs> accepted = solver.get_p();
  const VectorXs candidate0 = accepted[0] + VectorXs::Constant(1, Scalar(0.25));
  const VectorXs candidate1 = accepted[1] + VectorXs::Constant(2, Scalar(0.35));
  problem->update_p(candidate0, 0);
  problem->update_p(candidate1, 1);
  params_models[1]->calc(
      problem->get_paramsData()[1], params_models[1]->get_state()->zero(),
      VectorXs::Zero(params_models[1]->get_constraints()->get_nu()));
  const VectorXs candidate_h = problem->get_paramsData()[1]->constraints->h;
  BOOST_CHECK_NO_THROW(solver.computeDirection(true));
  BOOST_CHECK(problem->get_paramsData()[0]->params->params->p.isApprox(
      accepted[0], tol));
  BOOST_CHECK(problem->get_paramsData()[1]->params->params->p.isApprox(
      accepted[1], tol));
  BOOST_CHECK(
      !problem->get_paramsData()[1]->constraints->h.isApprox(candidate_h, tol));
  VectorXs reference(2);
  reference << Scalar(-0.2), Scalar(0.3);
  BOOST_CHECK(problem->get_paramsData()[1]->constraints->h.isApprox(
      accepted[1] - reference, tol));
}

template <typename Scalar>
void check_line_search_exception_rejects_candidate() {
  typedef crocoddyl::SolverFDDPTpl<Scalar> Solver;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  QuadraticParameterFixture<Scalar> fixture;
  CandidateExceptionSolver<Solver> solver(
      std::static_pointer_cast<typename Solver::ProblemAbstract>(
          fixture.problem));
  solver.solve(fixture.xs, fixture.us,
               std::vector<VectorXs>(1, fixture.initial), 1, true);
  BOOST_CHECK(!solver.accepted());
  solver.stopThrowing();
  solver.computeDirection(true);
  BOOST_CHECK(fixture.problem->get_paramsData()[0]->params->params->p.isApprox(
      fixture.initial));
  BOOST_CHECK(solver.get_p()[0].isApprox(fixture.initial));
}

template <typename Scalar>
void check_parameterized_intro_and_no_malloc() {
  typedef crocoddyl::SolverIntroTpl<Scalar> Solver;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  ParameterizedSolverFixture<Scalar> fixture;
  std::vector<VectorXs> init_p(2);
  init_p[0] = VectorXs::Constant(1, Scalar(0.15));
  init_p[1].resize(2);
  init_p[1] << Scalar(-0.2), Scalar(0.3);
  const crocoddyl::EqualitySolverType solvers[] = {
      crocoddyl::LuNull, crocoddyl::QrNull, crocoddyl::Schur};
  MatrixXs lu_basis;
  for (const crocoddyl::EqualitySolverType eq_solver : solvers) {
    Solver solver(fixture.problem, crocoddyl::FeasShoot, eq_solver,
                  crocoddyl::LuNull, crocoddyl::AStateNone);
    solver.solve(fixture.xs, fixture.us, init_p, 0, true);
    solver.computeDirection(true);
    const std::vector<
        std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>>& datas =
        fixture.problem->get_runningDatas();
    for (std::size_t t = 0; t < fixture.problem->get_T(); ++t) {
      BOOST_CHECK((datas[t]->Hu * solver.get_P()[t]).isApprox(datas[t]->Hp));
      if (eq_solver != crocoddyl::Schur) {
        const std::size_t nu =
            fixture.problem->get_runningModels()[t]->get_nu();
        const std::size_t nz = nu - solver.get_Hu_rank()[t];
        const std::size_t ndx = fixture.problem->get_ndx();
        const std::size_t nh_T =
            fixture.problem->get_terminalModel()->get_nh_T();
        BOOST_CHECK_EQUAL(solver.get_Qz()[t].size(), nz);
        BOOST_CHECK_EQUAL(solver.get_Qzz()[t].rows(), nz);
        BOOST_CHECK_EQUAL(solver.get_Qzz()[t].cols(), nz);
        BOOST_CHECK_EQUAL(solver.get_Qxz()[t].rows(), ndx);
        BOOST_CHECK_EQUAL(solver.get_Qxz()[t].cols(), nz);
        BOOST_CHECK_EQUAL(solver.get_Quz()[t].rows(), nu);
        BOOST_CHECK_EQUAL(solver.get_Quz()[t].cols(), nz);
        BOOST_CHECK_EQUAL(solver.get_Qzc()[t].rows(), nz);
        BOOST_CHECK_EQUAL(solver.get_Qzc()[t].cols(), nh_T);
        BOOST_CHECK((datas[t]->Hu * solver.get_YZ()[t].rightCols(nz))
                        .isZero(std::is_same<Scalar, float>::value
                                    ? Scalar(2e-5)
                                    : Scalar(1e-12)));
      }
    }
    if (eq_solver == crocoddyl::LuNull) {
      lu_basis = solver.get_YZ()[0];
    } else if (eq_solver == crocoddyl::QrNull) {
      BOOST_CHECK(!solver.get_YZ()[0].isApprox(
          lu_basis,
          std::is_same<Scalar, float>::value ? Scalar(2e-5) : Scalar(1e-12)));
    }

    const bool malloc_was_allowed =
        Eigen::internal::set_is_malloc_allowed(false);
    try {
      for (std::size_t i = 0; i < 100; ++i) {
        solver.computeDirection(true);
      }
      Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    } catch (...) {
      Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
      throw;
    }
    BOOST_CHECK_EQUAL(solver.get_dp()[0].size(), 1);
    BOOST_CHECK_EQUAL(solver.get_dp()[1].size(), 2);
  }

  Solver solver(fixture.problem, crocoddyl::FeasShoot, crocoddyl::Schur,
                crocoddyl::LuNull, crocoddyl::AStateNone);
  solver.solve(fixture.xs, fixture.us, init_p, 0, true);
  solver.set_equality_solver(crocoddyl::QrNull);
  solver.computeDirection(true);
  for (std::size_t t = 0; t < fixture.problem->get_T(); ++t) {
    const std::size_t nu = fixture.problem->get_runningModels()[t]->get_nu();
    const std::size_t nz = nu - solver.get_Hu_rank()[t];
    BOOST_CHECK_EQUAL(solver.get_Qz()[t].size(), nz);
    BOOST_CHECK_EQUAL(solver.get_Qzz()[t].rows(), nz);
    BOOST_CHECK_EQUAL(solver.get_Qzz()[t].cols(), nz);
  }
}

template <typename Scalar>
void check_impulse_node_solvers() {
  typedef crocoddyl::ActionModelAbstractTpl<Scalar> ActionModel;
  typedef crocoddyl::ContactModelTpl<Scalar> Contact;
  typedef crocoddyl::CostModelResidualTpl<Scalar> ResidualCost;
  typedef crocoddyl::CostModelSumTpl<Scalar> Costs;
  typedef crocoddyl::DiscretizedActionModelTpl<Scalar> ImpulseAction;
  typedef crocoddyl::DynamicsDataImpulseForwardTpl<Scalar> ImpulseData;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> ImpulseDynamics;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraints;
  typedef crocoddyl::ResidualModelStateTpl<Scalar> StateResidual;
  typedef crocoddyl::ShootingProblemTpl<Scalar> Problem;
  typedef crocoddyl::SolverFDDPTpl<Scalar> SolverFDDP;
  typedef crocoddyl::SolverIntroTpl<Scalar> SolverIntro;
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<crocoddyl::StateMultibody> state64 =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          StateModelFactory().create(StateModelTypes::StateMultibody_TalosArm));
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  const std::shared_ptr<ImplicitConstraints> contacts =
      std::make_shared<ImplicitConstraints>(state, 0);
  typename Contact::MaskArray mask = {{true, true, true, false, false, false}};
  const pinocchio::FrameIndex frame_id = static_cast<pinocchio::FrameIndex>(
      state->get_pinocchio()->frames.size() - 1);
  contacts->addConstraint(
      "contact",
      std::make_shared<Contact>(
          state, frame_id, pinocchio::SE3Tpl<Scalar>::Identity(),
          pinocchio::LOCAL_WORLD_ALIGNED, 0, Contact::Vector2s::Zero(), mask));
  const std::shared_ptr<ImpulseDynamics> dynamics =
      std::make_shared<ImpulseDynamics>(state, contacts);
  const std::shared_ptr<Costs> costs = std::make_shared<Costs>(state, 0);
  const std::shared_ptr<StateResidual> residual =
      std::make_shared<StateResidual>(state, 0);
  costs->addCost("state", std::make_shared<ResidualCost>(state, residual),
                 Scalar(1));
  const std::shared_ptr<ImpulseAction> impulse =
      std::make_shared<ImpulseAction>(dynamics, costs);
  const std::shared_ptr<Problem> problem = std::make_shared<Problem>(
      state->zero(), std::vector<std::shared_ptr<ActionModel>>(1, impulse),
      impulse);
  problem->set_nthreads(1);
  const std::vector<VectorXs> us(1, VectorXs::Zero(0));
  std::vector<VectorXs> xs(2);
  problem->rollout(us, xs);

  const std::shared_ptr<typename ImpulseAction::ActionDataAbstract>&
      action_data = problem->get_runningDatas()[0];
  const std::shared_ptr<typename ImpulseAction::Data> discretized_data =
      std::dynamic_pointer_cast<typename ImpulseAction::Data>(action_data);
  BOOST_REQUIRE(discretized_data != nullptr);
  const std::shared_ptr<ImpulseData> impulse_data =
      std::dynamic_pointer_cast<ImpulseData>(discretized_data->dynamics);
  BOOST_REQUIRE(impulse_data != nullptr);
  BOOST_CHECK_EQUAL(impulse_data->joint->tau.size(), 0);
  BOOST_CHECK_EQUAL(impulse_data->joint->dtau_dx.rows(), 0);
  BOOST_CHECK_EQUAL(impulse_data->joint->dtau_du.size(), 0);

  SolverFDDP fddp(problem);
  BOOST_CHECK_NO_THROW(fddp.solve(xs, us, std::vector<VectorXs>(), 1, true));
  BOOST_REQUIRE_EQUAL(fddp.get_us().size(), 1);
  BOOST_CHECK_EQUAL(fddp.get_us()[0].size(), 0);
  BOOST_CHECK(std::isfinite(static_cast<double>(fddp.get_cost())));

  SolverIntro intro(problem);
  BOOST_CHECK_NO_THROW(intro.solve(xs, us, std::vector<VectorXs>(), 1, true));
  BOOST_REQUIRE_EQUAL(intro.get_us().size(), 1);
  BOOST_CHECK_EQUAL(intro.get_us()[0].size(), 0);
  BOOST_CHECK(std::isfinite(static_cast<double>(intro.get_cost())));
}

template <typename Scalar>
void check_legacy_constructors_and_casts() {
  typedef crocoddyl::ActionModelAbstractTpl<Scalar> ActionModel;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> LQR;
  typedef crocoddyl::ProblemAbstractTpl<Scalar> ProblemAbstract;
  typedef crocoddyl::ShootingProblemTpl<Scalar> Problem;
  typedef crocoddyl::SolverFDDPTpl<Scalar> SolverFDDP;
  typedef crocoddyl::SolverIntroTpl<Scalar> SolverIntro;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  static_assert(std::is_constructible<SolverFDDP, std::shared_ptr<Problem>,
                                      crocoddyl::DynamicsSolverType,
                                      crocoddyl::EqualitySolverType>::value,
                "legacy SolverFDDP constructor must remain available");
  static_assert(
      std::is_constructible<
          SolverIntro, std::shared_ptr<Problem>, crocoddyl::DynamicsSolverType,
          crocoddyl::EqualitySolverType, crocoddyl::EqualitySolverType>::value,
      "legacy SolverIntro constructor must remain available");

  const std::shared_ptr<LQR> model = std::make_shared<LQR>(4, 2);
  const std::shared_ptr<LQR> terminal = std::make_shared<LQR>(4, 0);
  const std::shared_ptr<Problem> problem = std::make_shared<Problem>(
      VectorXs::Zero(4), std::vector<std::shared_ptr<ActionModel>>(2, model),
      terminal);
  SolverFDDP fddp(problem, crocoddyl::FeasShoot, crocoddyl::LuNull);
  SolverIntro intro(problem, crocoddyl::FeasShoot, crocoddyl::LuNull,
                    crocoddyl::LuNull);
  const auto fddp_cast = fddp.template cast<typename std::conditional<
      std::is_same<Scalar, double>::value, float, double>::type>();
  const auto intro_cast = intro.template cast<typename std::conditional<
      std::is_same<Scalar, double>::value, float, double>::type>();
  BOOST_CHECK_EQUAL(fddp_cast.get_problem()->get_T(), problem->get_T());
  BOOST_CHECK_EQUAL(intro_cast.get_problem()->get_T(), problem->get_T());

  const std::shared_ptr<ProblemAbstract> null_problem;
  BOOST_CHECK_THROW(SolverFDDP(null_problem, crocoddyl::FeasShoot,
                               crocoddyl::LuNull, crocoddyl::AStateNone),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      SolverIntro(null_problem, crocoddyl::FeasShoot, crocoddyl::LuNull,
                  crocoddyl::LuNull, crocoddyl::AStateNone),
      crocoddyl::Exception);
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_parameterized_solvers");
  ts->add(BOOST_TEST_CASE(&check_parameterized_fddp<double>));
  ts->add(BOOST_TEST_CASE(&check_parameterized_fddp<float>));
  ts->add(BOOST_TEST_CASE(&check_quadratic_parameter_fddp<double>));
  ts->add(BOOST_TEST_CASE(&check_quadratic_parameter_fddp<float>));
  ts->add(BOOST_TEST_CASE(&check_quadratic_parameter_intro<double>));
  ts->add(BOOST_TEST_CASE(&check_quadratic_parameter_intro<float>));
  ts->add(BOOST_TEST_CASE(&check_rejected_parameter_restoration<double>));
  ts->add(BOOST_TEST_CASE(&check_rejected_parameter_restoration<float>));
  ts->add(
      BOOST_TEST_CASE(&check_mixed_parameter_constraint_restoration<double>));
  ts->add(
      BOOST_TEST_CASE(&check_mixed_parameter_constraint_restoration<float>));
  ts->add(
      BOOST_TEST_CASE(&check_line_search_exception_rejects_candidate<double>));
  ts->add(
      BOOST_TEST_CASE(&check_line_search_exception_rejects_candidate<float>));
  ts->add(BOOST_TEST_CASE(&check_parameterized_intro_and_no_malloc<double>));
  ts->add(BOOST_TEST_CASE(&check_parameterized_intro_and_no_malloc<float>));
  ts->add(BOOST_TEST_CASE(&check_impulse_node_solvers<double>));
  ts->add(BOOST_TEST_CASE(&check_impulse_node_solvers<float>));
  ts->add(BOOST_TEST_CASE(&check_legacy_constructors_and_casts<double>));
  ts->add(BOOST_TEST_CASE(&check_legacy_constructors_and_casts<float>));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
