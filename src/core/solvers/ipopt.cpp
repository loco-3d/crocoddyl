#include "crocoddyl/core/solvers/ipopt.hpp"

namespace crocoddyl {

SolverIpopt::SolverIpopt(boost::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverAbstract(problem), ipopt_iface_(new IpoptInterface(problem)), app_(IpoptApplicationFactory()) {
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

bool SolverIpopt::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t maxiter, const bool is_feasible, const double regInit) {
  assert(init_xs.size() == ipopt_iface_->get_problem()->get_T() + 1);
  assert(init_us.size() == ipopt_iface_->get_problem()->get_T());

  if (init_xs != DEFAULT_VECTOR) {
    ipopt_iface_->set_xs(init_xs);
  }
  if (init_us != DEFAULT_VECTOR) {
    ipopt_iface_->set_us(init_us);
  }

  app_->Options()->SetIntegerValue("max_iter", maxiter);
  Ipopt::ApplicationReturnStatus status = app_->OptimizeTNLP(ipopt_iface_);

  std::copy(ipopt_iface_->get_xs().begin(), ipopt_iface_->get_xs().end(), xs_.begin());
  std::copy(ipopt_iface_->get_us().begin(), ipopt_iface_->get_us().end(), us_.begin());

  return status == Ipopt::Solve_Succeeded;
}

SolverIpopt::~SolverIpopt() {}

void SolverIpopt::computeDirection(const bool recalc) {}
double SolverIpopt::tryStep(const double steplength) { return 0.0; }
double SolverIpopt::stoppingCriteria() { return 0.0; }
const Eigen::Vector2d& SolverIpopt::expectedImprovement() { return Eigen::Vector2d::Zero(); }

}  // namespace crocoddyl