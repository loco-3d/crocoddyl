///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <example-robot-data/path.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/multibody/utils/quadruped-gaits.hpp"

int main(int argc, char* argv[]) {
  bool CALLBACKS = false;
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1;
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  pinocchio::Model model;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR
                              "/hyq_description/robots/hyq_no_sensors.urdf",
                              pinocchio::JointModelFreeFlyer(), model);
  pinocchio::srdf::loadReferenceConfigurations(
      model, EXAMPLE_ROBOT_DATA_MODEL_DIR "/hyq_description/srdf/hyq.srdf",
      false);

  crocoddyl::SimpleQuadrupedGaitProblem gait(model, "lf_foot", "rf_foot",
                                             "lh_foot", "rh_foot");

  const Eigen::VectorXd& x0 = gait.get_defaultState();

  // Walking gait_phase
  const double stepLength(0.25), stepHeight(0.25), timeStep(1e-2);
  const unsigned int stepKnots(25), supportKnots(2);

  // DDP Solver
  std::shared_ptr<crocoddyl::ShootingProblem> problem =
      gait.createWalkingProblem(x0, stepLength, stepHeight, timeStep, stepKnots,
                                supportKnots);
  crocoddyl::SolverFDDP solver(problem);
  if (CALLBACKS) {
    std::vector<std::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
    cbs.push_back(std::make_shared<crocoddyl::CallbackVerbose>());
    solver.setCallbacks(cbs);
  }

  // Initial State
  const std::size_t N = solver.get_problem()->get_T();
  std::vector<Eigen::VectorXd> xs(N, x0);
  std::vector<Eigen::VectorXd> us = problem->quasiStatic_xs(xs);
  xs.push_back(x0);

  std::cout << "NQ: "
            << solver.get_problem()->get_terminalModel()->get_state()->get_nq()
            << std::endl;
  std::cout << "Number of nodes: " << N << std::endl;

  // Solving the optimal control problem
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    solver.solve(xs, us, MAXITER, false, 0.1);
    duration[i] = timer.get_duration();
  }

  double avrg_duration = duration.mean();
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "  FDDP.solve [ms]: " << avrg_duration << " (" << min_duration
            << "-" << max_duration << ")" << std::endl;

  // Running calc
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    problem->calc(xs, us);
    duration[i] = timer.get_duration();
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calc [ms]: " << avrg_duration << " ("
            << min_duration << "-" << max_duration << ")" << std::endl;

  // Running calcDiff
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    problem->calcDiff(xs, us);
    duration[i] = timer.get_duration();
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calcDiff [ms]: " << avrg_duration << " ("
            << min_duration << "-" << max_duration << ")" << std::endl;
}
