#include "crocoddyl/core/activations/quadratic.hpp"

namespace crocoddyl {

ActivationModelQuad::ActivationModelQuad(const unsigned int& nr) : ActivationModelAbstract(nr) {}

ActivationModelQuad::~ActivationModelQuad() {}

void ActivationModelQuad::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& r) {
  assert(r.size() == nr_ && "ActivationModelQuad::calc: r has wrong dimension");
  data->a_value = 0.5 * r.transpose() * r;
}

void ActivationModelQuad::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  assert(r.size() == nr_ && "ActivationModelQuad::calcDiff: r has wrong dimension");
  if (recalc) {
    calc(data, r);
  }
  data->Ar = r;
  data->Arr.diagonal() = Eigen::VectorXd::Ones(nr_);
}

}  // namespace crocoddyl