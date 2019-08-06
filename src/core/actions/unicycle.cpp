#include "crocoddyl/core/actions/unicycle.hpp"

namespace crocoddyl {

ActionModelUnicycle::ActionModelUnicycle() : ActionModelAbstract(new StateVector(3), 2, 5), dt_(0.1) {
  cost_weights_ << 10., 1.;
}

ActionModelUnicycle::~ActionModelUnicycle() {
  // delete state_; //TODO @Carlos this breaks the test_actions c++ unit-test
}

void ActionModelUnicycle::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& x,
                               const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == nx_ && "ActionModelUnicycle::calc: x has wrong dimension");
  assert(u.size() == nu_ && "ActionModelUnicycle::calc: u has wrong dimension");

  ActionDataUnicycle* d = static_cast<ActionDataUnicycle*>(data.get());
  const double& c = std::cos(x[2]);
  const double& s = std::sin(x[2]);
  d->xnext << x[0] + c * u[0] * dt_, x[1] + s * u[0] * dt_, x[2] + u[1] * dt_;
  d->r.head<3>() = cost_weights_[0] * x;
  d->r.tail<2>() = cost_weights_[1] * u;
  d->cost = 0.5 * d->r.transpose() * d->r;
}

void ActionModelUnicycle::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& x,
                                   const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  assert(x.size() == nx_ && "ActionModelUnicycle::calcDiff: x has wrong dimension");
  assert(u.size() == nu_ && "ActionModelUnicycle::calcDiff: u has wrong dimension");

  if (recalc) {
    calc(data, x, u);
  }
  ActionDataUnicycle* d = static_cast<ActionDataUnicycle*>(data.get());

  // Cost derivatives
  const double& w_x = cost_weights_[0] * cost_weights_[0];
  const double& w_u = cost_weights_[1] * cost_weights_[1];
  d->Lx = x.cwiseProduct(Eigen::VectorXd::Constant(get_nx(), w_x));
  d->Lu = u.cwiseProduct(Eigen::VectorXd::Constant(get_nu(), w_u));
  d->Lxx.diagonal() << w_x, w_x, w_x;
  d->Luu.diagonal() << w_u, w_u;

  // Dynamic derivatives
  const double& c = std::cos(x[2]);
  const double& s = std::sin(x[2]);
  d->Fx << 1., 0., -s * u[0] * dt_, 0., 1., c * u[0] * dt_, 0., 0., 1.;
  d->Fu << c * dt_, 0., s * dt_, 0., 0., dt_;
}

boost::shared_ptr<ActionDataAbstract> ActionModelUnicycle::createData() {
  return boost::make_shared<ActionDataUnicycle>(this);
}

}  // namespace crocoddyl
