#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

DifferentialActionModelAbstract::DifferentialActionModelAbstract(const unsigned int& nq, const unsigned int& nv,
                                                                 const unsigned int& nu)
    : nq_(nq), nv_(nv), nu_(nu), nx_(nq + nv), ndx_(2 * nv), nout_(nv), ncost_(0), unone_(Eigen::VectorXd::Zero(nu)) {
  assert(nq_ != 0);
  assert(nv_ != 0);
  assert(nu_ != 0);
}

DifferentialActionModelAbstract::~DifferentialActionModelAbstract() {}

void DifferentialActionModelAbstract::calc(std::shared_ptr<DifferentialActionDataAbstract>& data,
                                           const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void DifferentialActionModelAbstract::calcDiff(std::shared_ptr<DifferentialActionDataAbstract>& data,
                                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

unsigned int DifferentialActionModelAbstract::get_nq() const { return nq_; }

unsigned int DifferentialActionModelAbstract::get_nv() const { return nv_; }

unsigned int DifferentialActionModelAbstract::get_nu() const { return nu_; }

unsigned int DifferentialActionModelAbstract::get_nx() const { return nx_; }

unsigned int DifferentialActionModelAbstract::get_ndx() const { return ndx_; }

unsigned int DifferentialActionModelAbstract::get_nout() const { return nout_; }

unsigned int DifferentialActionModelAbstract::get_ncost() const { return ncost_; }

}  // namespace crocoddyl