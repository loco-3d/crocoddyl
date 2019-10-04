#include "quadruped-gaits.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <crocoddyl/core/solvers/ddp.hpp>
#include <ctime>

int main(int argc, char* argv[]) {
  bool CALLBACKS = false;
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1000;

  if (argc > 1) {
    T = atoi(argv[1]);
  }
  
  pinocchio::Model rmodel;
  pinocchio::urdf::buildModel(HYQ_URDF, rmodel);
  crocoddyl::StateMultibody state(rmodel);

  crocoddyl::SimpleQuadrupedalGaitProblem gait(robot_model, "lfFoot", "rfFoot",
                                               "lhFoot", "rhFoot");

  const Eigen::VectorXd& x0 = gait.get_defaultState();

  //Walking gait_phase
  const double stepLength(0.25), stepHeight(0.25), timeStep(1e-2);
  const unsigned int  stepKnots (25), supportKnots(2);
  
  //DDP Solver
  crocoddyl::SolverDDP ddp(gait.createWalkingProblem(x0, stepLength, stepHeight, timeStep,
                                                     stepKnots, supportKnots));
  
  //Initial State
  const unsigned int N = ddp.get_problem().get_T();
  const std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(runningModel->get_nu()));
  for (unsigned int i = 0; i < N; ++i) {
    crocoddyl::ActionModelAbstract* model = problem.get_runningModels()[i];
    boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem.get_runningDatas()[i];
    model->quasiStatic(data, us[i], x0);
  }
  
  if (CALLBACKS) {
    std::vector<crocoddyl::CallbackAbstract*> cbs;
    cbs.push_back(new crocoddyl::CallbackVerbose());
    ddp.setCallbacks(cbs);
  }
  
  // Solving the optimal control problem
  struct timespec start, finish;
  double elapsed;
  Eigen::ArrayXd duration(T);
  Eigen::ArrayXi nIters(T);
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    ddp.solve(xs, us, MAXITER, false, 0.1);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = static_cast<double>(finish.tv_sec - start.tv_sec) * 1000000;
    elapsed += static_cast<double>(finish.tv_nsec - start.tv_nsec) / 1000;
    duration[i] = elapsed / 1000.;
    nIters[i] = ddp.get_iter();
  }

  double avrg_duration = duration.mean();
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "  DDP.solve [ms]: " << avrg_duration
            << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;
  std::cout << "  DDP.solve Mean Iter : " << nIters.mean() << std::endl;

  // Running calc
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    problem.calc(xs, us);
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
    problem.calcDiff(xs, us);
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
