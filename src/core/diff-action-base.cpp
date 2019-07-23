#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

DifferentialActionModelAbstract::DifferentialActionModelAbstract(StateAbstract* const state, const unsigned int& nu,
                                                                 const unsigned int& nr)
    : nq_(state->get_nq()),
      nv_(state->get_nv()),
      nu_(nu),
      nx_(state->get_nx()),
      ndx_(state->get_ndx()),
      nout_(state->get_nv()),
      nr_(nr),
      state_(state),
      unone_(Eigen::VectorXd::Zero(nu)) {
  assert(nq_ != 0);
  assert(nv_ != 0);
  assert(nu_ != 0);
}

DifferentialActionModelAbstract::~DifferentialActionModelAbstract() {}

void DifferentialActionModelAbstract::calc(boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                           const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void DifferentialActionModelAbstract::calcDiff(boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

unsigned int DifferentialActionModelAbstract::get_nq() const { return nq_; }

unsigned int DifferentialActionModelAbstract::get_nv() const { return nv_; }

unsigned int DifferentialActionModelAbstract::get_nu() const { return nu_; }

unsigned int DifferentialActionModelAbstract::get_nx() const { return nx_; }

unsigned int DifferentialActionModelAbstract::get_ndx() const { return ndx_; }

unsigned int DifferentialActionModelAbstract::get_nout() const { return nout_; }

unsigned int DifferentialActionModelAbstract::get_nr() const { return nr_; }

StateAbstract* DifferentialActionModelAbstract::get_state() const { return state_; }

}  // namespace crocoddyl