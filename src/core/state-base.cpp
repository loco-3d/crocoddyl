#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

StateAbstract::StateAbstract(const unsigned int& nx, const unsigned int& ndx) : nx_(nx), ndx_(ndx) {}

StateAbstract::~StateAbstract() {}

const unsigned int& StateAbstract::get_nx() const { return nx_; }

const unsigned int& StateAbstract::get_ndx() const { return ndx_; }

}  // namespace crocoddyl