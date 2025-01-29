///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2023, IRI: CSIC-UPC, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/ipopt.hpp"

namespace crocoddyl {

SolverIpopt::SolverIpopt(std::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverAbstract(problem),
      ipopt_iface_(new IpoptInterface(problem)),
      ipopt_app_(IpoptApplicationFactory()) {
  ipopt_app_->Options()->SetNumericValue("tol", th_stop_);
  ipopt_app_->Options()->SetStringValue("mu_strategy", "adaptive");

  ipopt_status_ = ipopt_app_->Initialize();

  if (ipopt_status_ != Ipopt::Solve_Succeeded) {
    std::cerr << "Error during IPOPT initialization!" << std::endl;
  }
}

bool SolverIpopt::solve(const std::vector<Eigen::VectorXd>& init_xs,
                        const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t maxiter, const bool is_feasible,
                        const double /*reg_init*/) {
  setCandidate(init_xs, init_us, is_feasible);
  ipopt_iface_->set_xs(xs_);
  ipopt_iface_->set_us(us_);

  ipopt_app_->Options()->SetIntegerValue("max_iter",
                                         static_cast<Ipopt::Index>(maxiter));
  ipopt_status_ = ipopt_app_->OptimizeTNLP(ipopt_iface_);

  std::copy(ipopt_iface_->get_xs().begin(), ipopt_iface_->get_xs().end(),
            xs_.begin());
  std::copy(ipopt_iface_->get_us().begin(), ipopt_iface_->get_us().end(),
            us_.begin());
  cost_ = ipopt_iface_->get_cost();
  iter_ = ipopt_app_->Statistics()->IterationCount();

  return ipopt_status_ == Ipopt::Solve_Succeeded ||
         ipopt_status_ == Ipopt::Solved_To_Acceptable_Level;
}

void SolverIpopt::resizeData() {
  SolverAbstract::resizeData();
  ipopt_iface_->resizeData();
}

SolverIpopt::~SolverIpopt() {}

void SolverIpopt::computeDirection(const bool) {}

double SolverIpopt::tryStep(const double) { return 0.; }

double SolverIpopt::stoppingCriteria() { return 0.; }

const Eigen::Vector2d& SolverIpopt::expectedImprovement() { return d_; }

void SolverIpopt::setStringIpoptOption(const std::string& tag,
                                       const std::string& value) {
  ipopt_app_->Options()->SetStringValue(tag, value);
}

void SolverIpopt::setNumericIpoptOption(const std::string& tag,
                                        Ipopt::Number value) {
  ipopt_app_->Options()->SetNumericValue(tag, value);
}

void SolverIpopt::set_th_stop(const double th_stop) {
  if (th_stop <= 0.) {
    throw_pretty("Invalid argument: " << "th_stop value has to higher than 0.");
  }
  th_stop_ = th_stop;
  ipopt_app_->Options()->SetNumericValue("tol", th_stop_);
}

}  // namespace crocoddyl
