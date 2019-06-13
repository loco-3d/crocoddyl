#include <crocoddyl/core/solver-base.hpp>

namespace crocoddyl {

SolverAbstract::SolverAbstract(ShootingProblem& problem) : problem(problem), isFeasible(false),
    cost(0.), stop(0.), x_reg(NAN), u_reg(NAN), stepLength(1.), dV(0.), dV_exp(0.),
    th_acceptStep(0.1), th_stop(1e-9), iter(0) {}

SolverAbstract::~SolverAbstract() {}

void SolverAbstract::setCandidate(const std::vector<Eigen::VectorXd>& xs_warm,
                                  const std::vector<Eigen::VectorXd>& us_warm,
                                  const bool& _isFeasible) {
  const long unsigned int& T = problem.get_T();

  if (xs_warm.size() == 0) {
    for (long unsigned int t = 0; t < T; ++t) {
      xs[t] = problem.runningModels[t]->get_state()->zero();
    }
    xs.back() = problem.terminalModel->get_state()->zero();
  } else {
    assert(xs_warm.size()==T+1);
    std::copy(xs_warm.begin(), xs_warm.end(), xs.begin());
  }

  if(us_warm.size() == 0) {
    for (long unsigned int t = 0; t < T; ++t) {
      const int& nu = problem.runningModels[t]->get_nu();
      us[t] = Eigen::VectorXd::Zero(nu);
    }
  } else {
    assert(us_warm.size()==T);
    std::copy(us_warm.begin(), us_warm.end(), us.begin());
  }
  isFeasible = _isFeasible;
}

void SolverAbstract::setCallbacks(std::vector<CallbackAbstract*>& _callbacks) {
  callbacks = _callbacks;
}

const bool& SolverAbstract::get_isFeasible() const {
  return isFeasible;
}

const unsigned int& SolverAbstract::get_iter() const {
  return iter;
}

const double& SolverAbstract::get_cost() const {
  return cost;
}

const double& SolverAbstract::get_stop() const {
  return stop;
}

const Eigen::Vector2d& SolverAbstract::get_d() const {
  return d;
}

const double& SolverAbstract::get_Xreg() const {
  return x_reg;
}

const double& SolverAbstract::get_Ureg() const {
  return u_reg;
}

const double& SolverAbstract::get_stepLength() const {
  return stepLength;
}

const double& SolverAbstract::get_dV() const {
  return dV;
}

const double& SolverAbstract::get_dVexp() const {
  return dV_exp;
}

}  // namespace crocoddyl