#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include <time.h>

#ifdef WITH_MULTITHREADING
#include <omp.h>
#endif  // WITH_MULTITHREADING

int main() {
  bool CALLBACKS = false;
  unsigned int N = 200;  // number of nodes
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1;
  using namespace crocoddyl;

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<ActionModelAbstract*> runningModels;
  ActionModelAbstract* terminalModel;
  x0 = Eigen::Vector3d(1., 0., 0.);

  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < N; ++i) {
    ActionModelAbstract* model_i = new ActionModelUnicycle();
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::Vector2d::Zero());
  }
  xs.push_back(x0);
  terminalModel = new ActionModelUnicycle();

  // Formulating the optimal control problem
  ShootingProblem problem(x0, runningModels, terminalModel);
  SolverDDP ddp(problem);
  if (CALLBACKS) {
    std::vector<CallbackAbstract*> cbs;
    cbs.push_back(new CallbackVerbose());
    ddp.setCallbacks(cbs);
  }
  struct timespec start, finish;
  double elapsed;
  // Solving the optimal control problem
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    ddp.solve(xs, us, MAXITER);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec) * 1000000.0;
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000.0;

    duration[i] = elapsed;  // in us
  }

  double avrg_duration = duration.sum() / T;
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "Wall time solve [us]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;

  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    problem.calcDiff(xs, us);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec) * 1000000.0;
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000.0;
    duration[i] = elapsed;  // in us
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "Wall time calcDiff [us]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;
}
