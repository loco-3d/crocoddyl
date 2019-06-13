#include <crocoddyl/core/actions/unicycle.hpp>

namespace crocoddyl {

ActionModelUnicycle::ActionModelUnicycle(StateAbstract *const state) : ActionModelAbstract(state, 2), ncost(5),
    dt(0.1) {
  costWeights << 10., 1.;
}

ActionModelUnicycle::~ActionModelUnicycle() {}

void ActionModelUnicycle::calc(std::shared_ptr<ActionDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& x,
                               const Eigen::Ref<const Eigen::VectorXd>& u) {
  DataUnicycle* d = static_cast<DataUnicycle*>(data.get());
  const double& c = std::cos(x[2]);
  const double& s = std::sin(x[2]);
  d->xnext << x[0] + c * u[0] * dt,
              x[1] + s * u[0] * dt,
              x[2] + u[1] * dt;
  d->costResiduals.head<3>() = costWeights[0] * x;
  d->costResiduals.tail<2>() = costWeights[1] * u;
  d->cost = 0.5 * d->costResiduals.transpose() * d->costResiduals;
}

void ActionModelUnicycle::calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& x,
                                   const Eigen::Ref<const Eigen::VectorXd>& u,
                                   const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  DataUnicycle* d = static_cast<DataUnicycle*>(data.get());

  // Cost derivatives
  const double& w_x = costWeights[0] * costWeights[0];
  const double& w_u = costWeights[1] * costWeights[1];
  d->Lx = x.cwiseProduct(Eigen::VectorXd::Constant(get_nx(), w_x));
  d->Lu = u.cwiseProduct(Eigen::VectorXd::Constant(get_nu(), w_u));
  d->Lxx.diagonal() << w_x, w_x, w_x;
  d->Luu.diagonal() << w_u, w_u;

  // Dynamic derivatives
  const double& c = std::cos(x[2]);
  const double& s = std::sin(x[2]);
  d->Fx << 1., 0., -s * u[0] * dt,
           0., 1., c * u[0] * dt,
           0., 0., 1.;
  d->Fu << c * dt, 0.,
           s * dt, 0.,
           0., dt;
}

std::shared_ptr<ActionDataAbstract> ActionModelUnicycle::createData() {
  return std::make_shared<DataUnicycle>(this);
}

unsigned int ActionModelUnicycle::get_ncost() const {
  return ncost;
}

}  // namespace crocoddyl
