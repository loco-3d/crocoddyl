#include <crocoddyl/core/optctrl/shooting.hpp>

namespace crocoddyl {

ShootingProblem::ShootingProblem(const Eigen::Ref<const Eigen::VectorXd>& x0,
                                 std::vector<ActionModelAbstract*>& runningModels,
                                 ActionModelAbstract* terminalModel) : terminalModel(terminalModel),
    runningModels(runningModels), T(runningModels.size()), x0(x0), cost(0.) {
  for (unsigned int i = 0; i < runningModels.size(); ++i) {
    ActionModelAbstract* model = runningModels[i];
    runningDatas.push_back(model->createData());
  }
  terminalData = terminalModel->createData();
}

ShootingProblem::~ShootingProblem() {}

double ShootingProblem::calc(const std::vector<Eigen::VectorXd>& xs,
                             const std::vector<Eigen::VectorXd>& us) {
  double cost = 0;
  for (unsigned int i = 0; i < T; ++i) {
    ActionModelAbstract* model = runningModels[i];
    std::shared_ptr<ActionDataAbstract>& data = runningDatas[i];
    const Eigen::VectorXd& x = xs[i+1];
    const Eigen::VectorXd& u = us[i];

    model->calc(data, x, u);
    cost += data->cost;
  }
  terminalModel->calc(terminalData, xs.back());
  cost += terminalData->cost;
  return cost;
}

double ShootingProblem::calcDiff(const std::vector<Eigen::VectorXd>& xs,
                                 const std::vector<Eigen::VectorXd>& us) {
  cost = 0;
  for (long unsigned int i = 0; i < T; ++i) {
    ActionModelAbstract* model = runningModels[i];
    std::shared_ptr<ActionDataAbstract>& data = runningDatas[i];
    const Eigen::VectorXd& x = xs[i];
    const Eigen::VectorXd& u = us[i];

    model->calcDiff(data, x, u);
    cost += data->cost;
  }
  terminalModel->calcDiff(terminalData, xs.back());
  cost += terminalData->cost;
  return cost;
}

void ShootingProblem::rollout(const std::vector<Eigen::VectorXd>& us,
                              std::vector<Eigen::VectorXd>& xs) {
  xs.resize(T+1);
  xs[0] = x0;
  for (long unsigned int i = 0; i < T; ++i) {
    ActionModelAbstract* model = runningModels[i];
    std::shared_ptr<ActionDataAbstract>& data = runningDatas[i];
    Eigen::VectorXd& x = xs[i];
    const Eigen::VectorXd& u = us[i];

    model->calc(data, x, u);
    xs[i+1] = data->get_xnext();
  }
}

long unsigned int ShootingProblem::get_T() const {
  return T;
}

Eigen::VectorXd& ShootingProblem::get_x0() {
  return x0;
}

}  // namespace crocoddyl