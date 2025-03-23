///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"

int main(int argc, char* argv[]) {
  unsigned int NX = 37;
  unsigned int NU = 12;
  bool CALLBACKS = false;
  unsigned int N = 100;  // number of nodes
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1;
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  // Creating the action models and warm point for the LQR system
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(NX);
  std::shared_ptr<crocoddyl::ActionModelAbstract> model =
      std::make_shared<crocoddyl::ActionModelLQR>(NX, NU);
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(NU));
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(
      N, model);

  // Formulating the optimal control problem
  std::shared_ptr<crocoddyl::ShootingProblem> problem =
      std::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, model);
  crocoddyl::SolverFDDP solver(problem);
  if (CALLBACKS) {
    std::vector<std::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
    cbs.push_back(std::make_shared<crocoddyl::CallbackVerbose>());
    solver.setCallbacks(cbs);
  }

  std::cout << "NQ: "
            << solver.get_problem()->get_terminalModel()->get_state()->get_nq()
            << std::endl;
  std::cout << "Number of nodes: " << N << std::endl;

  // Solving the optimal control problem
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    solver.solve(xs, us, MAXITER);
    duration[i] = timer.get_duration();
  }

  double avrg_duration = duration.sum() / T;
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
