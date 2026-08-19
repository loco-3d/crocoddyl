///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          INRIA, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/intro.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "factory/solver.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//____________________________________________________________________________//

namespace {

Eigen::VectorXd sampleSolverState(
    const std::shared_ptr<crocoddyl::StateAbstract>& state,
    const double tangent_scale = 1.) {
  return sampleUnitTestState(state, tangent_scale);
}

Eigen::VectorXd perturbSolverState(
    const std::shared_ptr<crocoddyl::StateAbstract>& state,
    const Eigen::VectorXd& x, const double tangent_scale = 1e-1) {
  Eigen::VectorXd dx = tangent_scale * random_vector<double>(state->get_ndx());
  Eigen::VectorXd x_perturbed = x;
  state->integrate(x, dx, x_perturbed);
  return x_perturbed;
}

Eigen::VectorXd sampleSolverControl(const std::size_t nu,
                                    const double scale = 1.) {
  return scale * random_vector<double>(static_cast<Eigen::Index>(nu));
}

void sampleSolverTrajectoryGuess(
    const std::shared_ptr<crocoddyl::ProblemAbstract>& problem,
    std::vector<Eigen::VectorXd>* xs, std::vector<Eigen::VectorXd>* us,
    const double x_scale = 1., const double u_scale = 1.) {
  xs->clear();
  us->clear();
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      problem->get_runningModels()[0]->get_state();
  const std::size_t T = problem->get_T();
  xs->reserve(T + 1);
  us->reserve(T);
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
        problem->get_runningModels()[i];
    xs->push_back(sampleSolverState(state, x_scale));
    us->push_back(sampleSolverControl(model->get_nu(), u_scale));
  }
  xs->push_back(sampleSolverState(state, x_scale));
}

void perturbSolverTrajectoryGuess(
    const std::shared_ptr<crocoddyl::ProblemAbstract>& problem,
    std::vector<Eigen::VectorXd>* xs, std::vector<Eigen::VectorXd>* us,
    const double x_scale = 1e-1, const double u_scale = 1e-1) {
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      problem->get_runningModels()[0]->get_state();
  const std::size_t T = problem->get_T();
  for (std::size_t i = 0; i < T; ++i) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
        problem->get_runningModels()[i];
    (*xs)[i] = perturbSolverState(state, (*xs)[i], x_scale);
    (*us)[i] += sampleSolverControl(model->get_nu(), u_scale);
  }
  xs->back() = perturbSolverState(state, xs->back(), x_scale);
}

}  // namespace

//____________________________________________________________________________//

void test_solver_compute_direction(SolverTypes::Type solver_type,
                                   ActionModelTypes::Type action_type,
                                   size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the testing and KKT solvers
  SolverFactory solver_factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      solver_factory.create(solver_type, model, model2, modelT, T);

  // Get the pointer to the problem so we can create the equivalent kkt solver.
  const std::shared_ptr<crocoddyl::ProblemAbstract>& problem =
      solver->get_problem();

  // Generate the different state along the trajectory
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);

  // Compute search direction
  solver->set_preg(random_real_in_range<double>(10., 100.));
  solver->setCandidate(xs, us);
  solver->computeDirection(true);

  // Check that Vxx is a symmetric matrix
  std::vector<Eigen::MatrixXd> Vxx =
      solver_factory.get_Vxx(solver_type, solver);
  std::vector<Eigen::VectorXd> Vx = solver_factory.get_Vx(solver_type, solver);
  for (std::size_t i = 0; i < problem->get_T() + 1; ++i) {
    BOOST_CHECK(Vxx[i].isApprox(Vxx[i].transpose()));
  }

  // Check the search direction and Lagrange multiplier for nodes
  const std::size_t ndx = problem->get_ndx();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * ndx, 2 * ndx);
  Eigen::VectorXd a = Eigen::VectorXd::Zero(2 * ndx);
  A.topRightCorner(ndx, ndx).diagonal().array() = -1.;
  A.bottomLeftCorner(ndx, ndx).diagonal().array() = -1.;
  for (std::size_t i = 0; i < problem->get_T(); ++i) {
    const Eigen::VectorXd& f = solver->get_fs()[i];
    A.bottomRightCorner(ndx, ndx) = Vxx[i];
    a.head(ndx) = f;
    a.tail(ndx) = Vx[i];
    const Eigen::VectorXd& w = -A.inverse() * a;
    BOOST_CHECK(isCloseAbsRel(w.head(ndx), (Vx[i] + Vxx[i] * f), 1e-9, 1e-9));
    BOOST_CHECK(isCloseAbsRel(w.tail(ndx), f, 1e-9, 1e-9));
  }
  // Check the Schur-complement of terminal-constraint direction is Hermite
  if (problem->get_terminalModel()->get_nh_T() != 0) {
    const Eigen::MatrixXd& dHc = solver_factory.get_dHc(solver_type, solver);
    // Check that dHc is a symmetric matrix
    BOOST_CHECK(dHc.isApprox(dHc.transpose()));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(dHc);
    const Eigen::VectorXd& eigenvalues = eigen_solver.eigenvalues();
    // Check that all real parts of the eigenvalues are positive
    for (int i = 0; i < eigenvalues.size(); ++i) {
      BOOST_CHECK_GT(eigenvalues[i], 0.0);
    }
  }
}

//____________________________________________________________________________//

void test_solver_gaps_evolution(SolverTypes::Type solver_type,
                                ActionModelTypes::Type action_type, size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the testing and KKT solvers
  SolverFactory solver_factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      solver_factory.create(solver_type, model, model2, modelT, T);

  // Get the pointer to the problem so we can create the equivalent kkt solver.
  const std::shared_ptr<crocoddyl::ProblemAbstract>& problem =
      solver->get_problem();

  // Generate the different state along the trajectory
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);

  // One step optimization with alpha=1. This case the dynamics gaps are fully
  // closed.
  double alpha = 1;
  solver->setCandidate(xs, us);
  solver->computeDirection(true);
  solver->expectedImprovement();
  solver->tryStep(alpha);
  for (unsigned int t = 0; t < T + 1; ++t) {
    BOOST_CHECK(solver->get_fs_try()[t].isZero(1e-9));
  }
  // Check that the terminal constraints is fully satisfied with alpha=1
  if (solver->get_problem()->get_terminalModel()->get_nh_T() != 0) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        problem->get_terminalData();
    BOOST_CHECK(data->h.isZero(1e-9));
  }

  // One step optimization with a random alpha. This case the dynamics gaps are
  // closed by (1 - alpha) factor.
  if (!SolverTypes::isSingleShoot(solver_type)) {
    alpha = random_real_in_range<double>(0.25, 0.75);
    solver->computeDirection(true);
    solver->expectedImprovement();
    solver->tryStep(alpha);
    for (unsigned int t = 0; t < T + 1; ++t) {
      BOOST_CHECK(isCloseAbsRel(solver->get_fs_try()[t],
                                (1 - alpha) * solver->get_fs()[t], 1e-9, 1e-9));
    }
    // Check that the terminal constraints is fully satisfied with alpha=1
    if (solver->get_problem()->get_terminalModel()->get_nh_T() != 0) {
      const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
          problem->get_terminalModel();
      const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
          problem->get_terminalData();
      const std::shared_ptr<crocoddyl::ActionDataAbstract>& data2 =
          model->createData();
      model->calc(data2, solver->get_xs().back());
      BOOST_CHECK(isCloseAbsRel(data->h, (1 - alpha) * data2->h, 1e-9, 1e-9));
    }
  }

  // One step optimization with a random alpha and regularization. This case the
  // dynamics gaps are closed by (1 - alpha) factor.
  if (!SolverTypes::isSingleShoot(solver_type)) {
    alpha = random_real_in_range<double>(0.25, 0.75);
    solver->set_preg(random_real_in_range<double>(10., 100.));
    solver->computeDirection(true);
    solver->expectedImprovement();
    solver->tryStep(alpha);
    for (unsigned int t = 0; t < T + 1; ++t) {
      BOOST_CHECK(isCloseAbsRel(solver->get_fs_try()[t],
                                (1 - alpha) * solver->get_fs()[t], 1e-9, 1e-9));
    }
  } else {
    for (unsigned int t = 0; t < T + 1; ++t) {
      BOOST_CHECK(solver->get_fs_try()[t].isZero(1e-9));
    }
  }
}

//____________________________________________________________________________//

void test_solver_expected_improvement(SolverTypes::Type solver_type,
                                      ActionModelTypes::Type action_type,
                                      size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the testing and KKT solvers
  SolverFactory solver_factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      solver_factory.create(solver_type, model, model2, modelT, T);

  // Get the pointer to the problem so we can create the equivalent kkt solver.
  const std::shared_ptr<crocoddyl::ProblemAbstract>& problem =
      solver->get_problem();

  // Generate the different state along the trajectory
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);

  // One step optimization with alpha=1.
  double alpha = 1;
  solver->computeDirection(true);
  solver->expectedImprovement();
  solver->tryStep(alpha);
  BOOST_CHECK_CLOSE(solver->get_dVexp(), solver->get_dV(), 1e-9);

  // One step optimization with a random alpha. This case the dynamics gaps are
  // closed by (1 - alpha) factor.
  alpha = random_real_in_range<double>(0.25, 0.75);
  solver->tryStep(alpha);
  BOOST_CHECK_CLOSE(solver->get_dVexp(), solver->get_dV(), 1e-9);

  // One step optimization with a random alpha and regularization. This case the
  // dynamics gaps are closed by (1 - alpha) factor.
  if (!SolverTypes::isSingleShoot(solver_type)) {
    alpha = random_real_in_range<double>(0.25, 0.75);
    solver->set_preg(random_real_in_range<double>(10., 100.));
    solver->computeDirection(true);
    solver->expectedImprovement();
    solver->tryStep(alpha);
    BOOST_CHECK_CLOSE(solver->get_dVexp(), solver->get_dV(), 1e-9);
  }

  // One step optimization when closing the gaps with random alpha and
  // regularization
  solver->get_problem()->rollout(us, xs);
  solver->setCandidate(xs, us);
  solver->computeDynamicFeasibility();
  alpha = random_real_in_range<double>(0.25, 0.75);
  solver->computeDirection(true);
  solver->expectedImprovement();
  solver->tryStep(alpha);
  BOOST_CHECK_CLOSE(solver->get_dVexp(), solver->get_dV(), 1e-9);
}

//____________________________________________________________________________//

void test_solver_convergence(SolverTypes::Type solver_type,
                             ActionModelTypes::Type action_type, size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the testing and KKT solvers
  SolverFactory solver_factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      solver_factory.create(solver_type, model, model2, modelT, T);

  // Get the pointer to the problem so we can create the equivalent kkt solver.
  const std::shared_ptr<crocoddyl::ProblemAbstract>& problem =
      solver->get_problem();

  // Generate the different state along the trajectory
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);

  // // Define the callback function
  // std::vector<std::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
  // cbs.push_back(std::make_shared<crocoddyl::CallbackVerbose>());

  // // Print the name of the action model for introspection
  // solver->setCallbacks(cbs);
  // std::cout << ActionModelTypes::all[action_type] << std::endl;

  // Compute solve and check expected convergence and step lenght
  solver->solve();
  BOOST_CHECK_EQUAL(solver->get_iter(), 1);
  BOOST_CHECK_EQUAL(solver->get_steplength(), 1.);

  // Compute solve with random warmstarting and check expected convergence and
  // step length
  xs = solver->get_xs();
  us = solver->get_us();
  perturbSolverTrajectoryGuess(problem, &xs, &us);
  solver->solve(xs, us);
  BOOST_CHECK_EQUAL(solver->get_iter(), 1);
  BOOST_CHECK_EQUAL(solver->get_steplength(), 1.);

  // Compute solve with regularization and check expected convergence
  perturbSolverTrajectoryGuess(problem, &xs, &us);
  solver->solve(xs, us, 10, false, random_real_in_range<double>(10., 100.));
  BOOST_CHECK_EQUAL(solver->get_steplength(), 1.);
}

//____________________________________________________________________________//

void test_casted_solver(SolverTypes::Type solver_type,
                        ActionModelTypes::Type action_type, size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the solver
  SolverFactory solver_factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      solver_factory.create(solver_type, model, model2, modelT, T);
  std::shared_ptr<crocoddyl::SolverAbstractTpl<float>> casted_solver =
      solver->cast<float>();

  // Get the pointer to the problem so we can create the equivalent kkt solver.
  const std::shared_ptr<crocoddyl::ProblemAbstract>& problem =
      solver->get_problem();

  // Generate the different state along the trajectory
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);
  std::vector<Eigen::VectorXf> xs_f = crocoddyl::vector_cast<float>(xs);
  std::vector<Eigen::VectorXf> us_f = crocoddyl::vector_cast<float>(us);

  // Compute solve with coldstart
  float tol_f = 80.f * std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  solver->solve();
  casted_solver->solve();
  for (std::size_t i = 0; i < T; ++i) {
    BOOST_CHECK((solver->get_xs()[i].cast<float>() - casted_solver->get_xs()[i])
                    .isZero(tol_f));
    BOOST_CHECK((solver->get_us()[i].cast<float>() - casted_solver->get_us()[i])
                    .isZero(tol_f));
  }
  BOOST_CHECK(
      (solver->get_xs().back().cast<float>() - casted_solver->get_xs().back())
          .isZero(tol_f));

  // Compute solve with warmstart
  solver->solve(xs, us);
  casted_solver->solve(xs_f, us_f);
  for (std::size_t i = 0; i < T; ++i) {
    BOOST_CHECK((solver->get_xs()[i].cast<float>() - casted_solver->get_xs()[i])
                    .isZero(tol_f));
    BOOST_CHECK((solver->get_us()[i].cast<float>() - casted_solver->get_us()[i])
                    .isZero(tol_f));
  }
  BOOST_CHECK(
      (solver->get_xs().back().cast<float>() - casted_solver->get_xs().back())
          .isZero(tol_f));
}

//____________________________________________________________________________//

class SolverFDDPArrivalStateProbe : public crocoddyl::SolverFDDP {
 public:
  using crocoddyl::SolverFDDP::SolverFDDP;

  void set_initial_direction(const Eigen::VectorXd& dx0) { dxs_[0] = dx0; }

  void set_no_improvement_reject_case() {
    dPhi_ = 0.;
    dPhiexp_ = 0.;
    dV_ = 0.;
    dVexp_ = 1.;
    DV_.setZero();
    DV_[1] = 10. * get_th_grad();
    dImpr_ = 0.;
    steplength_ = 1.;
  }

#ifdef CROCODDYL_WITH_ODYN
  void run_astate_qp_semidefinite_initial_value_case() {
    BOOST_REQUIRE_EQUAL(n_phases_, 1);
    BOOST_REQUIRE_EQUAL(p_.size(), 1);
    BOOST_REQUIRE_GT(p_[0].size(), 0);
    BOOST_REQUIRE(!Vxx_.empty());

    Vpp_phase_[0].setIdentity();
    Vp_phase_[0].setZero();
    Vpx_phase_[0].setZero();
    Vpx_f_phase_[0].setZero();
    Vxx_[0].setZero();
    Vx_[0].setZero();
    dxs_[0].setZero();
    for (std::size_t t = 0; t < P_.size(); ++t) {
      P_[t].setZero();
      k_[t].setZero();
    }

    paramsPass();

    BOOST_CHECK(dp_[0].isZero(1e-12));
  }
#endif
};

void test_fddp_legacy_solve_api() {
  typedef crocoddyl::ActionModelLQR ActionModelLQR;

  const std::size_t T = 2;
  const std::shared_ptr<ActionModelLQR> model =
      std::make_shared<ActionModelLQR>(4, 2);
  const std::shared_ptr<ActionModelLQR> terminal_model =
      std::make_shared<ActionModelLQR>(4, 0);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      T, model);
  const Eigen::VectorXd x0 = sampleSolverState(model->get_state());
  const std::shared_ptr<crocoddyl::ShootingProblem> problem =
      std::make_shared<crocoddyl::ShootingProblem>(x0, running_models,
                                                   terminal_model);
  crocoddyl::SolverFDDP solver(problem);

  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);
  BOOST_CHECK_NO_THROW(solver.solve(xs, us, 0, false, 0.1));
}

void test_fddp_single_shoot_arrival_state_forward_pass() {
  typedef crocoddyl::ActionModelLQR ActionModelLQR;
  typedef crocoddyl::LQRParams LQRParams;
  typedef crocoddyl::ParameterManager ParameterManager;
  typedef crocoddyl::ShootingProblem ShootingProblem;

  const std::size_t T = 2;
  const std::shared_ptr<ActionModelLQR> model =
      std::make_shared<ActionModelLQR>(4, 2, 1, 0, 0, true);
  const std::shared_ptr<ActionModelLQR> modelT =
      std::make_shared<ActionModelLQR>(4, 0, 1, 0, 0, true);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(model->get_state());
  params->addParam("lqr", std::make_shared<LQRParams>(model->get_state(), 1));
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      T, model);
  const std::shared_ptr<crocoddyl::StateAbstract>& state = model->get_state();
  const Eigen::VectorXd x0 = sampleSolverState(state);
  const std::shared_ptr<ShootingProblem> problem =
      std::make_shared<ShootingProblem>(
          x0, running_models, modelT,
          std::make_shared<crocoddyl::ParameterPhaseModel>(params));
  problem->set_nthreads(1);
  SolverFDDPArrivalStateProbe solver(
      problem, crocoddyl::DynamicsSolverType::SingleShoot,
      crocoddyl::EqualitySolverType::LuNull,
      crocoddyl::ArrivalStateSolverType::AStateLuNull);

  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);
  solver.setCandidate(xs, us);
  const double alpha = random_real_in_range<double>(0.25, 0.75);
  const Eigen::VectorXd dx0 =
      1e-1 * random_vector<double>(static_cast<Eigen::Index>(state->get_ndx()));
  solver.set_initial_direction(dx0);
  Eigen::VectorXd expected = state->zero();
  state->integrate(solver.get_xs()[0], alpha * dx0, expected);

  solver.singleShootForwardPass(alpha);

  BOOST_CHECK((state->diff_dx(expected, solver.get_xs_try()[0])).isZero(1e-12));

  const std::shared_ptr<ActionModelLQR> standard_model =
      std::make_shared<ActionModelLQR>(4, 2);
  const std::shared_ptr<ActionModelLQR> standard_terminal =
      std::make_shared<ActionModelLQR>(4, 0);
  const std::shared_ptr<crocoddyl::ShootingProblem> standard_problem =
      std::make_shared<crocoddyl::ShootingProblem>(
          x0,
          std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>(
              T, standard_model),
          standard_terminal);
  SolverFDDPArrivalStateProbe standard_solver(
      std::static_pointer_cast<crocoddyl::ProblemAbstract>(standard_problem),
      crocoddyl::DynamicsSolverType::SingleShoot,
      crocoddyl::EqualitySolverType::LuNull,
      crocoddyl::ArrivalStateSolverType::AStateLuNull);
  std::vector<Eigen::VectorXd> standard_xs;
  std::vector<Eigen::VectorXd> standard_us;
  sampleSolverTrajectoryGuess(standard_problem, &standard_xs, &standard_us);
  standard_solver.setCandidate(standard_xs, standard_us);
  standard_solver.set_initial_direction(dx0);
  standard_solver.singleShootForwardPass(alpha);
  BOOST_CHECK(
      (state->diff_dx(x0, standard_solver.get_xs_try()[0])).isZero(1e-12));

  // An arrival state is a decision variable, so it must not be counted as an
  // initial dynamics defect. Ordinary problems preserve the legacy behavior.
  xs[0] = perturbSolverState(state, x0);
  for (std::size_t t = 0; t < T; ++t) {
    model->calc(problem->get_runningDatas()[t], xs[t], us[t]);
    xs[t + 1] = problem->get_runningDatas()[t]->xnext;
  }
  solver.setCandidate(xs, us);
  solver.calcDir();
  BOOST_CHECK(solver.get_fs()[0].isZero(1e-12));
  BOOST_CHECK_SMALL(solver.get_ffeas(), 1e-12);

  standard_xs[0] = xs[0];
  for (std::size_t t = 0; t < T; ++t) {
    standard_model->calc(standard_problem->get_runningDatas()[t],
                         standard_xs[t], standard_us[t]);
    standard_xs[t + 1] = standard_problem->get_runningDatas()[t]->xnext;
  }
  standard_solver.setCandidate(standard_xs, standard_us);
  standard_solver.calcDir();
  BOOST_CHECK(!standard_solver.get_fs()[0].isZero(1e-12));
  BOOST_CHECK_GT(standard_solver.get_ffeas(), 0.);
}

void test_fddp_zero_control_running_dispatch() {
  const std::size_t T = 2;
  const std::size_t nx = 4;
  const std::shared_ptr<crocoddyl::ActionModelLQR> model =
      std::make_shared<crocoddyl::ActionModelLQR>(nx, 0);
  const std::shared_ptr<crocoddyl::ActionModelLQR> terminal_model =
      std::make_shared<crocoddyl::ActionModelLQR>(nx, 0);
  const Eigen::VectorXd x0 = sampleSolverState(model->get_state());
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>
      running_models(T, model);
  const std::shared_ptr<crocoddyl::ShootingProblem> problem =
      std::make_shared<crocoddyl::ShootingProblem>(x0, running_models,
                                                   terminal_model);
  problem->set_nthreads(1);
  SolverFDDPArrivalStateProbe solver(problem);
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);

  const crocoddyl::DynamicsSolverType types[] = {
      crocoddyl::FeasShoot, crocoddyl::MultiShoot, crocoddyl::HybridShoot,
      crocoddyl::SingleShoot};
  for (std::size_t i = 0; i < 4; ++i) {
    solver.set_dynamics_solver(types[i],
                               types[i] == crocoddyl::HybridShoot ? 1 : 0);
    solver.setCandidate(xs, us);
    solver.computeDirection(true);
    switch (types[i]) {
      case crocoddyl::FeasShoot:
        solver.feasShootForwardPass(0.5);
        break;
      case crocoddyl::MultiShoot:
        solver.multiShootForwardPass(0.5);
        break;
      case crocoddyl::HybridShoot:
        solver.hybridShootForwardPass(0.5);
        break;
      case crocoddyl::SingleShoot:
        solver.singleShootForwardPass(0.5);
        break;
    }
    const std::shared_ptr<crocoddyl::ActionDataAbstract> expected =
        model->createData();
    model->calc(expected, solver.get_xs_try()[0], us[0]);
    BOOST_CHECK(
        model->get_state()
            ->diff_dx(expected->xnext, problem->get_runningDatas()[0]->xnext)
            .isZero(1e-12));
  }
}

void test_fddp_check_acceptance_preserves_legacy_no_improvement() {
  ActionModelFactory action_factory;
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      action_factory.create(ActionModelTypes::ActionModelLQR);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      action_factory.create(ActionModelTypes::ActionModelLQR,
                            ActionModelFactory::Terminal);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      1, model);
  const Eigen::VectorXd x0 = sampleSolverState(model->get_state());
  std::shared_ptr<crocoddyl::ShootingProblem> problem =
      std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, modelT);
  SolverFDDPArrivalStateProbe solver(problem);

  solver.set_no_improvement_reject_case();

  BOOST_CHECK(solver.checkAcceptance());
}

void test_fddp_parametrized_arrival_state_no_malloc() {
  typedef crocoddyl::ActionModelLQR ActionModelLQR;
  typedef crocoddyl::LQRParams LQRParams;
  typedef crocoddyl::ParameterManager ParameterManager;
  typedef crocoddyl::ShootingProblem ShootingProblem;

  const std::size_t nx = 8;
  const std::size_t nu = 4;
  const std::size_t np = 3;
  const std::size_t T = 4;
  const std::shared_ptr<ActionModelLQR> model =
      std::make_shared<ActionModelLQR>(nx, nu, np);
  const std::shared_ptr<ActionModelLQR> terminal_model =
      std::make_shared<ActionModelLQR>(nx, 0, np);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(model->get_state());
  params->addParam("lqr", std::make_shared<LQRParams>(model->get_state(), np));

  const Eigen::VectorXd x0 = sampleSolverState(model->get_state());
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      T, model);
  const std::shared_ptr<ShootingProblem> problem =
      std::make_shared<ShootingProblem>(
          x0, running_models, terminal_model,
          std::make_shared<crocoddyl::ParameterPhaseModel>(params));
  problem->set_nthreads(1);

  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);
  const Eigen::VectorXd p = params->rand();
  std::vector<Eigen::VectorXd> init_p(1, p);

  SolverFDDPArrivalStateProbe solver(
      problem, crocoddyl::DynamicsSolverType::FeasShoot,
      crocoddyl::EqualitySolverType::LuNull,
      crocoddyl::ArrivalStateSolverType::AStateLuNull);
  solver.set_no_improvement_reject_case();
  BOOST_CHECK(!solver.checkAcceptance());
  solver.solve(xs, us, init_p, 0, false);

  solver.computeDirection(true);
  const bool malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    solver.computeDirection(true);
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  BOOST_REQUIRE_EQUAL(solver.get_p().size(), 1);
  BOOST_CHECK(solver.get_p()[0].isApprox(p, 1e-12));
}

#ifdef CROCODDYL_WITH_ODYN
void test_fddp_astate_qp_skips_initial_state_marginalization() {
  typedef crocoddyl::ActionModelLQR ActionModelLQR;
  typedef crocoddyl::LQRParams LQRParams;
  typedef crocoddyl::ParameterManager ParameterManager;
  typedef crocoddyl::ShootingProblem ShootingProblem;

  const std::size_t nx = 8;
  const std::size_t nu = 4;
  const std::size_t np = 3;
  const std::size_t T = 2;
  const std::shared_ptr<ActionModelLQR> model =
      std::make_shared<ActionModelLQR>(nx, nu, np);
  const std::shared_ptr<ActionModelLQR> terminal_model =
      std::make_shared<ActionModelLQR>(nx, 0, np);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(model->get_state());
  params->addParam("lqr", std::make_shared<LQRParams>(model->get_state(), np));

  const Eigen::VectorXd x0 = sampleSolverState(model->get_state());
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      T, model);
  const std::shared_ptr<ShootingProblem> problem =
      std::make_shared<ShootingProblem>(
          x0, running_models, terminal_model,
          std::make_shared<crocoddyl::ParameterPhaseModel>(params));
  problem->set_nthreads(1);

  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);
  const Eigen::VectorXd p = params->rand();
  std::vector<Eigen::VectorXd> init_p(1, p);

  SolverFDDPArrivalStateProbe solver(
      problem, crocoddyl::DynamicsSolverType::FeasShoot,
      crocoddyl::EqualitySolverType::LuNull,
      crocoddyl::ArrivalStateSolverType::AStateQP);
  solver.solve(xs, us, init_p, 0, false);
  problem->update_p(p, 0);

  solver.run_astate_qp_semidefinite_initial_value_case();
}
#endif

void test_intro_parametrized_equality_no_malloc() {
  typedef crocoddyl::ActionModelLQR ActionModelLQR;
  typedef crocoddyl::LQRParams LQRParams;
  typedef crocoddyl::ParameterManager ParameterManager;
  typedef crocoddyl::ShootingProblem ShootingProblem;

  const std::size_t nx = 8;
  const std::size_t nu = 4;
  const std::size_t np = 3;
  const std::size_t nh = 2;
  const std::size_t ng = 0;
  const std::size_t T = 4;

  const Eigen::MatrixXd A = random_matrix<double>(
      static_cast<Eigen::Index>(nx), static_cast<Eigen::Index>(nx));
  const Eigen::MatrixXd B = random_matrix<double>(
      static_cast<Eigen::Index>(nx), static_cast<Eigen::Index>(nu));
  const Eigen::MatrixXd P = random_matrix<double>(
      static_cast<Eigen::Index>(nx), static_cast<Eigen::Index>(np));
  const Eigen::MatrixXd M =
      random_matrix<double>(static_cast<Eigen::Index>(nx + nu + np),
                            static_cast<Eigen::Index>(nx + nu + np));
  const Eigen::MatrixXd L =
      M.transpose() * M + Eigen::MatrixXd::Identity(nx + nu + np, nx + nu + np);
  const Eigen::MatrixXd Q = L.topLeftCorner(nx, nx);
  const Eigen::MatrixXd N = L.block(0, nx, nx, nu);
  const Eigen::MatrixXd Y = L.topRightCorner(nx, np);
  const Eigen::MatrixXd R = L.block(nx, nx, nu, nu);
  const Eigen::MatrixXd V = L.block(nx, nx + nu, nu, np);
  const Eigen::MatrixXd W = L.bottomRightCorner(np, np);
  const Eigen::MatrixXd G = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(ng), static_cast<Eigen::Index>(nx + nu + np));
  Eigen::MatrixXd H = random_matrix<double>(
      static_cast<Eigen::Index>(nh), static_cast<Eigen::Index>(nx + nu + np));
  H.middleCols(static_cast<Eigen::Index>(nx), static_cast<Eigen::Index>(nh))
      .diagonal()
      .array() += 1.;
  const Eigen::VectorXd f = random_vector<double>(nx);
  const Eigen::VectorXd q = random_vector<double>(nx);
  const Eigen::VectorXd r = random_vector<double>(nu);
  const Eigen::VectorXd m = random_vector<double>(np);
  const Eigen::VectorXd g = Eigen::VectorXd::Zero(ng);
  const Eigen::VectorXd h = random_vector<double>(nh);

  const std::shared_ptr<ActionModelLQR> model =
      std::make_shared<ActionModelLQR>(A, B, P, Q, R, N, W, Y, V, G, H, f, q, r,
                                       m, g, h);
  const std::shared_ptr<ActionModelLQR> terminal_model =
      std::make_shared<ActionModelLQR>(nx, 0, np);
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(model->get_state());
  params->addParam("lqr", std::make_shared<LQRParams>(model->get_state(), np));

  const Eigen::VectorXd x0 = sampleSolverState(model->get_state());
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      T, model);
  const std::shared_ptr<ShootingProblem> problem =
      std::make_shared<ShootingProblem>(
          x0, running_models, terminal_model,
          std::make_shared<crocoddyl::ParameterPhaseModel>(params));
  problem->set_nthreads(1);

  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);
  const Eigen::VectorXd p = params->rand();
  std::vector<Eigen::VectorXd> init_p(1, p);

  crocoddyl::SolverIntro solver(problem,
                                crocoddyl::DynamicsSolverType::FeasShoot,
                                crocoddyl::EqualitySolverType::LuNull,
                                crocoddyl::EqualitySolverType::LuNull,
                                crocoddyl::ArrivalStateSolverType::AStateNone);
  solver.solve(xs, us, init_p, 0, false);

  solver.computeDirection(true);
  const bool malloc_was_allowed = Eigen::internal::set_is_malloc_allowed(false);
  try {
    solver.computeDirection(true);
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  BOOST_REQUIRE_EQUAL(solver.get_p().size(), 1);
  BOOST_CHECK(solver.get_p()[0].isApprox(p, 1e-12));
}

//____________________________________________________________________________//

void test_kkt_dimension(ActionModelTypes::Type action_type, size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the kkt solver
  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverKKT> kkt =
      std::static_pointer_cast<crocoddyl::SolverKKT>(
          factory.create(SolverTypes::SolverKKT, model, model2, modelT, T));

  // define some aliases
  const std::size_t ndx = kkt->get_ndx();
  const std::size_t nu = kkt->get_nu();

  // Test the different matrix sizes
  BOOST_CHECK_EQUAL(kkt->get_kkt().rows(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt->get_kkt().cols(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt->get_kktref().size(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt->get_primaldual().size(), 2 * ndx + nu);
  BOOST_CHECK_EQUAL(kkt->get_us().size(), T);
  BOOST_CHECK_EQUAL(kkt->get_xs().size(), T + 1);
}

//____________________________________________________________________________//

void test_kkt_search_direction(ActionModelTypes::Type action_type, size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the kkt solver
  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverKKT> kkt =
      std::static_pointer_cast<crocoddyl::SolverKKT>(
          factory.create(SolverTypes::SolverKKT, model, model2, modelT, T));

  // Generate the different state along the trajectory
  const std::shared_ptr<crocoddyl::ProblemAbstract>& problem =
      kkt->get_problem();
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);

  // Compute the search direction
  kkt->setCandidate(xs, us);
  kkt->computeDirection();

  // define some aliases
  const std::size_t ndx = kkt->get_ndx();
  const std::size_t nu = kkt->get_nu();
  Eigen::MatrixXd kkt_mat = kkt->get_kkt();
  Eigen::Block<Eigen::MatrixXd> hess = kkt_mat.block(0, 0, ndx + nu, ndx + nu);

  // Checking the symmetricity of the Hessian
  BOOST_CHECK(isCloseAbsRel(hess, hess.transpose(), 1e-9, 1e-9));

  // Check initial state
  BOOST_CHECK((problem->get_runningModels()[0]->get_state()->diff_dx(
                   problem->get_runningModels()[0]->get_state()->integrate_x(
                       xs[0], kkt->get_dxs()[0]),
                   kkt->get_problem()->get_x0()))
                  .isZero(1e-9));
}

//____________________________________________________________________________//

void test_solver_against_reference_solver(SolverTypes::Type solver_type,
                                          ActionModelTypes::Type action_type,
                                          size_t T) {
  // Create action models
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      ActionModelFactory().create(action_type);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model2 =
      ActionModelFactory().create(action_type, ActionModelFactory::Second);
  std::shared_ptr<crocoddyl::ActionModelAbstract> modelT =
      ActionModelFactory().create(action_type, ActionModelFactory::Terminal);

  // Create the testing solver and reference solver
  SolverFactory solver_factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      solver_factory.create(solver_type, model, model2, modelT, T);
  std::shared_ptr<crocoddyl::SolverAbstract> reference = solver_factory.create(
      SolverTypes::getReferenceSolverType(), model, model2, modelT, T);

  // Get the pointer to the problem so we can create the equivalent reference
  // solver.
  const std::shared_ptr<crocoddyl::ProblemAbstract>& problem =
      solver->get_problem();
  const std::shared_ptr<crocoddyl::StateAbstract>& state =
      problem->get_runningModels()[0]->get_state();

  // Generate the different state along the trajectory
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  sampleSolverTrajectoryGuess(problem, &xs, &us);

  // Define the callback function
  std::vector<std::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
  cbs.push_back(std::make_shared<crocoddyl::CallbackVerbose>());

  // Print the name of the action model for introspection
  // kkt->setCallbacks(cbs);
  // solver->setCallbacks(cbs);
  // std::cout << ActionModelTypes::all[action_type] << std::endl;

  // Solve the problem using the reference solver
  reference->solve(xs, us, 100);

  // Solve the problem using the solver in testing
  solver->solve(xs, us, 100);

  // check trajectory dimensions
  BOOST_CHECK_EQUAL(solver->get_us().size(), T);
  BOOST_CHECK_EQUAL(solver->get_xs().size(), T + 1);

  // initial state
  BOOST_CHECK((solver->get_xs()[0] - problem->get_x0()).isZero(1e-9));

  // check solutions against each other
  for (unsigned int t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
        solver->get_problem()->get_runningModels()[t];
    std::size_t nu = model->get_nu();
    BOOST_CHECK((state->diff_dx(solver->get_xs()[t], reference->get_xs()[t]))
                    .isZero(1e-9));
    BOOST_CHECK(
        (solver->get_us()[t].head(nu) - reference->get_us()[t]).isZero(1e-9));
  }
  BOOST_CHECK((state->diff_dx(solver->get_xs()[T], reference->get_xs()[T]))
                  .isZero(1e-9));
}

//____________________________________________________________________________//

void register_solvers_againt_lqr_actions_unit_tests(
    SolverTypes::Type solver_type, ActionModelTypes::Type action_type,
    const std::size_t T) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << solver_type << "_against_lqr_action_" << action_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_solver_compute_direction,
                                      solver_type, action_type, T)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_solver_gaps_evolution, solver_type, action_type, T)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_solver_expected_improvement,
                                      solver_type, action_type, T)));
  if (solver_type != SolverTypes::SolverBoxFDDP_SingleShoot &&
      solver_type != SolverTypes::SolverBoxFDDP_FeasShoot &&
      solver_type != SolverTypes::SolverBoxFDDP_MultiShoot &&
      solver_type != SolverTypes::SolverBoxFDDP_HybridShoot &&
      solver_type != SolverTypes::SolverIpopt) {
    ts->add(BOOST_TEST_CASE(
        boost::bind(&test_solver_convergence, solver_type, action_type, T)));
  }
  if ((solver_type != SolverTypes::SolverIpopt) &&
      (solver_type != SolverTypes::SolverKKT)) {
    ts->add(BOOST_TEST_CASE(
        boost::bind(&test_casted_solver, solver_type, action_type, T)));
  }
  framework::master_test_suite().add(ts);
}

//____________________________________________________________________________//

void register_kkt_solver_unit_tests(ActionModelTypes::Type action_type,
                                    const std::size_t T) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_SolverKKT_" << action_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_kkt_dimension, action_type, T)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_kkt_search_direction, action_type, T)));
  framework::master_test_suite().add(ts);
}

void register_solvers_against_reference_unit_tests(
    SolverTypes::Type solver_type, ActionModelTypes::Type action_type,
    const std::size_t T) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << solver_type << "_vs_"
            << SolverTypes::getReferenceSolverName() << "_" << action_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_solver_against_reference_solver,
                                      solver_type, action_type, T)));
  framework::master_test_suite().add(ts);
}

void register_fddp_arrival_state_unit_tests() {
  test_suite* ts =
      BOOST_TEST_SUITE("test_SolverFDDP_single_shoot_arrival_state");
  std::cout << "Running test_SolverFDDP_single_shoot_arrival_state"
            << std::endl;
  ts->add(BOOST_TEST_CASE(&test_fddp_legacy_solve_api));
  ts->add(BOOST_TEST_CASE(&test_fddp_single_shoot_arrival_state_forward_pass));
  ts->add(BOOST_TEST_CASE(&test_fddp_zero_control_running_dispatch));
  ts->add(BOOST_TEST_CASE(
      &test_fddp_check_acceptance_preserves_legacy_no_improvement));
  ts->add(BOOST_TEST_CASE(&test_fddp_parametrized_arrival_state_no_malloc));
#ifdef CROCODDYL_WITH_ODYN
  ts->add(BOOST_TEST_CASE(
      &test_fddp_astate_qp_skips_initial_state_marginalization));
#endif
  framework::master_test_suite().add(ts);
}

void register_intro_no_malloc_unit_tests() {
  test_suite* ts = BOOST_TEST_SUITE("test_SolverIntro_no_malloc");
  std::cout << "Running test_SolverIntro_no_malloc" << std::endl;
  ts->add(BOOST_TEST_CASE(&test_intro_parametrized_equality_no_malloc));
  framework::master_test_suite().add(ts);
}

//____________________________________________________________________________//

bool init_function() {
#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(1);
#endif
  setenv("OMP_DYNAMIC", "FALSE", 1);
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("KMP_ALL_THREADS", "1", 1);

  std::size_t T = 10;

  for (size_t s = 1; s < SolverTypes::all.size(); ++s) {
    for (size_t i = ActionModelTypes::ActionModelLQRDriftFree;
         i < ActionModelTypes::NbActionModelTypes; ++i) {
      if (SolverTypes::all[s] != SolverTypes::SolverIpopt) {
#ifdef CROCODDYL_WITH_ODYN
        if (SolverTypes::all[s] != SolverTypes::SolverOdynSQP) {
          register_solvers_againt_lqr_actions_unit_tests(
              SolverTypes::all[s], ActionModelTypes::all[i], T);
        }
#else
        register_solvers_againt_lqr_actions_unit_tests(
            SolverTypes::all[s], ActionModelTypes::all[i], T);
#endif
      }
    }
  }

  for (size_t i = 0; i < ActionModelTypes::all.size(); ++i) {
    register_kkt_solver_unit_tests(ActionModelTypes::all[i], T);
  }

  // We start from 1 as 0 is the kkt solver
  for (size_t s = 1; s < SolverTypes::all.size(); ++s) {
    for (size_t i = ActionModelTypes::ActionModelLQRDriftFree;
         i < ActionModelTypes::NbActionModelTypes; ++i) {
      if (SolverTypes::all[s] != SolverTypes::getReferenceSolverType() &&
          !ActionModelTypes::hasTerminalConstraints(ActionModelTypes::all[i])) {
        register_solvers_against_reference_unit_tests(
            SolverTypes::all[s], ActionModelTypes::all[i], T);
      }
    }
  }
  register_fddp_arrival_state_unit_tests();
  register_intro_no_malloc_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
