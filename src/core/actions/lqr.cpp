#include <crocoddyl/core/actions/lqr.hpp>

namespace crocoddyl {

ActionModelLQR::ActionModelLQR(const unsigned int& nx,
                               const unsigned int& nu,
															 bool driftFree) : ActionModelAbstract(new StateVector(nx), nu),
		driftFree(driftFree) {
  //TODO substitute by random (vectors) and random-orthogonal (matrices)
  Fx = Eigen::MatrixXd::Identity(nx, nx);
  Fu = Eigen::MatrixXd::Identity(nx, nu);
  f0 = Eigen::VectorXd::Ones(nx);
  Lxx = Eigen::MatrixXd::Identity(nx, nx);
  Lxu = Eigen::MatrixXd::Identity(nx, nu);
  Luu = Eigen::MatrixXd::Identity(nu, nu);
  lx = Eigen::VectorXd::Ones(nx);
  lu = Eigen::VectorXd::Ones(nu);
}

ActionModelLQR::~ActionModelLQR() {}

void ActionModelLQR::calc(std::shared_ptr<ActionDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>& x,
                          const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (driftFree) {
    data->xnext = Fx * x + Fu * u;
  } else {
    data->xnext = Fx * x + Fu * u + f0;
  }
  data->cost = 0.5 * x.dot(Lxx * x) + 0.5 * u.dot(Luu * u) + x.dot(Lxu * u) + lx.dot(x) + lu.dot(u);
}

void ActionModelLQR::calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x,
                              const Eigen::Ref<const Eigen::VectorXd>& u,
                              const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  data->Lx = lx + Lxx * x + Lxu * u;
  data->Lu = lu + Lxu.transpose() * x + Luu * u;
  data->Fx = Fx;
  data->Fu = Fu;
  data->Lxx = Lxx;
  data->Lxu = Lxu;
  data->Luu = Luu;
}

std::shared_ptr<ActionDataAbstract> ActionModelLQR::createData() {
  return std::make_shared<ActionDataLQR>(this);
}

}  // namespace crocoddyl