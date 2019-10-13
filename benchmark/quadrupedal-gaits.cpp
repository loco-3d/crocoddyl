///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <ctime>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include "crocoddyl/multibody/utils/quadruped-gaits.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

int main(int argc, char* argv[]) {
  bool CALLBACKS = false;
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1;
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  pinocchio::Model rmodel;
  pinocchio::urdf::buildModel(HYQ_URDF, pinocchio::JointModelFreeFlyer(), rmodel);
  pinocchio::srdf::loadReferenceConfigurations(rmodel, HYQ_SRDF, false);

  crocoddyl::SimpleQuadrupedGaitProblem gait(rmodel, "lf_foot", "rf_foot", "lh_foot", "rh_foot");

  const Eigen::VectorXd& x0 = gait.get_defaultState();

  // Walking gait_phase
  const double stepLength(0.25), stepHeight(0.25), timeStep(1e-2);
  const unsigned int stepKnots(25), supportKnots(2);

  // DDP Solver
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      gait.createWalkingProblem(x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots);
  crocoddyl::SolverFDDP ddp(problem);
  if (CALLBACKS) {
    std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
    cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
    ddp.setCallbacks(cbs);
  }

  // Initial State
  const std::size_t& N = ddp.get_problem()->get_T();
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(problem->get_runningModels()[0]->get_nu()));
  for (std::size_t i = 0; i < N; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem->get_runningDatas()[i];
    model->quasiStatic(data, us[i], x0);
  }

  // Solving the optimal control problem
  struct timespec start, finish;
  double elapsed;
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    ddp.solve(xs, us, MAXITER, false, 0.1);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = static_cast<double>(finish.tv_sec - start.tv_sec) * 1000000;
    elapsed += static_cast<double>(finish.tv_nsec - start.tv_nsec) / 1000;
    duration[i] = elapsed / 1000.;
  }

  double avrg_duration = duration.mean();
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "  DDP.solve [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;

  // Running calc
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    problem->calc(xs, us);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = static_cast<double>(finish.tv_sec - start.tv_sec) * 1000000;
    elapsed += static_cast<double>(finish.tv_nsec - start.tv_nsec) / 1000;
    duration[i] = elapsed / 1000.;
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calc [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;

  // Running calcDiff
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    problem->calcDiff(xs, us);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = static_cast<double>(finish.tv_sec - start.tv_sec) * 1000000;
    elapsed += static_cast<double>(finish.tv_nsec - start.tv_nsec) / 1000;
    duration[i] = elapsed / 1000.;
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calcDiff [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration
            << ")" << std::endl;
}