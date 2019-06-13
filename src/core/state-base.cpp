#include <crocoddyl/core/state-base.hpp>

namespace crocoddyl {

StateAbstract::StateAbstract(const unsigned int& nx,
                             const unsigned int& ndx) : nx(nx), ndx(ndx) {}

StateAbstract::~StateAbstract() {}

const unsigned int& StateAbstract::get_nx() const {
  return nx;
}

const unsigned int& StateAbstract::get_ndx() const {
  return ndx;
}

}  // namespace crocoddyl