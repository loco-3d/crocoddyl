#include "crocoddyl/core/solvers/ipopt/ipopt.hpp"

namespace crocoddyl {

SolverIpOpt::SolverIpOpt(boost::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverAbstract(problem), ms_nlp_(new MultipleShootingNlp(problem)), app_(IpoptApplicationFactory()) {
  app_->Options()->SetNumericValue("tol", 3.82e-6);
  app_->Options()->SetStringValue("mu_strategy", "adaptive");
  // app_->Options()->SetStringValue("max_iter", 100);
  // app->Options()->SetStringValue("output_file", "ipopt.out");

  status_ = app_->Initialize();

  if (status_ != Ipopt::Solve_Succeeded) {
    // Throw an exception here
    std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
  }
}

bool SolverIpOpt::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t maxiter, const bool is_feasible, const double regInit) {
  assert(init_xs.size() == ms_nlp_->get_problem()->get_T() + 1);
  assert(init_us.size() == ms_nlp_->get_problem()->get_T());

  if (init_xs != DEFAULT_VECTOR) {
    ms_nlp_->set_xs(init_xs);
  }
  if (init_us != DEFAULT_VECTOR) {
    ms_nlp_->set_us(init_us);
  }

  app_->Options()->SetIntegerValue("max_iter", maxiter);
  Ipopt::ApplicationReturnStatus status = app_->OptimizeTNLP(ms_nlp_);

  std::copy(ms_nlp_->get_xs().begin(), ms_nlp_->get_xs().end(), xs_.begin());
  std::copy(ms_nlp_->get_us().begin(), ms_nlp_->get_us().end(), us_.begin());

  return status == Ipopt::Solve_Succeeded;
}

SolverIpOpt::~SolverIpOpt() {}

void SolverIpOpt::computeDirection(const bool recalc) {}
double SolverIpOpt::tryStep(const double steplength) { return 0.0; }
double SolverIpOpt::stoppingCriteria() { return 0.0; }
const Eigen::Vector2d& SolverIpOpt::expectedImprovement() { return Eigen::Vector2d::Zero(); }

}  // namespace crocoddyl