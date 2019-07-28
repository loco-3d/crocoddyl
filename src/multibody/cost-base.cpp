#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"

namespace crocoddyl {

CostModelAbstract::CostModelAbstract(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                                     const unsigned int& nu, const bool& with_residuals)
    : pinocchio_(model),
      activation_(activation),
      nq_(model->nq),
      nv_(model->nv),
      nu_(nu),
      nx_(model->nq + model->nv),
      ndx_(2 * model->nv),
      nr_(activation->get_nr()),
      with_residuals_(with_residuals),
      unone_(Eigen::VectorXd::Zero(nu)) {
  assert(nq_ != 0 && "CostModelAbstract: nq cannot be 0");
  assert(nv_ != 0 && "CostModelAbstract: nv cannot be 0");
}

CostModelAbstract::CostModelAbstract(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                                     const bool& with_residuals)
    : pinocchio_(model),
      activation_(activation),
      nq_(model->nq),
      nv_(model->nv),
      nu_(model->nv),
      nx_(model->nq + model->nv),
      ndx_(2 * model->nv),
      nr_(activation->get_nr()),
      with_residuals_(with_residuals),
      unone_(Eigen::VectorXd::Zero(model->nv)) {
  assert(nq_ != 0 && "CostModelAbstract: nq cannot be 0");
  assert(nv_ != 0 && "CostModelAbstract: nv cannot be 0");
}

CostModelAbstract::CostModelAbstract(pinocchio::Model* const model, const unsigned int& nr, const unsigned int& nu,
                                     const bool& with_residuals)
    : pinocchio_(model),
      activation_(new ActivationModelQuad(nr)),
      nq_(model->nq),
      nv_(model->nv),
      nu_(nu),
      nx_(model->nq + model->nv),
      ndx_(2 * model->nv),
      nr_(nr),
      with_residuals_(with_residuals),
      unone_(Eigen::VectorXd::Zero(nu)) {
  assert(nq_ != 0 && "CostModelAbstract: nq cannot be 0");
  assert(nv_ != 0 && "CostModelAbstract: nv cannot be 0");
}

CostModelAbstract::CostModelAbstract(pinocchio::Model* const model, const unsigned int& nr, const bool& with_residuals)
    : pinocchio_(model),
      activation_(new ActivationModelQuad(nr)),
      nq_(model->nq),
      nv_(model->nv),
      nu_(model->nv),
      nx_(model->nq + model->nv),
      ndx_(2 * model->nv),
      nr_(nr),
      with_residuals_(with_residuals),
      unone_(Eigen::VectorXd::Zero(model->nv)) {
  assert(nq_ != 0 && "CostModelAbstract: nq cannot be 0");
  assert(nv_ != 0 && "CostModelAbstract: nv cannot be 0");
}

CostModelAbstract::~CostModelAbstract() {}

void CostModelAbstract::calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void CostModelAbstract::calcDiff(boost::shared_ptr<CostDataAbstract>& data,
                                 const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

boost::shared_ptr<CostDataAbstract> CostModelAbstract::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataAbstract>(this, data);
}

pinocchio::Model* CostModelAbstract::get_pinocchio() const { return pinocchio_; }

ActivationModelAbstract* CostModelAbstract::get_activation() const { return activation_; }

const unsigned int& CostModelAbstract::get_nq() const { return nq_; }

const unsigned int& CostModelAbstract::get_nv() const { return nv_; }

const unsigned int& CostModelAbstract::get_nu() const { return nu_; }

const unsigned int& CostModelAbstract::get_nx() const { return nx_; }

const unsigned int& CostModelAbstract::get_ndx() const { return ndx_; }

const unsigned int& CostModelAbstract::get_nr() const { return nr_; }

}  // namespace crocoddyl