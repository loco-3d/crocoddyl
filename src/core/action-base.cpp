#include <crocoddyl/core/action-base.hpp>

namespace crocoddyl {

//struct ActionDataAbstract; // forward declaration

ActionModelAbstract::ActionModelAbstract(StateAbstract *const state,
                                         const unsigned int& nu) : nx(state->get_nx()),
    ndx(state->get_ndx()), nu(nu), state(state), unone(Eigen::VectorXd::Zero(nu)) {
  assert(nx != 0);
  assert(ndx != 0);
  assert(nu != 0);
}

ActionModelAbstract::~ActionModelAbstract() {}

void ActionModelAbstract::calc(std::shared_ptr<ActionDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone);
}

void ActionModelAbstract::calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone);
}

unsigned int ActionModelAbstract::get_nx() const {
  return nx;
}

unsigned int ActionModelAbstract::get_ndx() const {
  return ndx;
}

unsigned int ActionModelAbstract::get_nu() const {
  return nu;
}

StateAbstract* ActionModelAbstract::get_state() const {
  return state;
}

}  // namespace crocoddyl