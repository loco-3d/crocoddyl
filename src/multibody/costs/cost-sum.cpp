#include "crocoddyl/multibody/costs/cost-sum.hpp"

namespace crocoddyl {

CostModelSum::CostModelSum(pinocchio::Model* const model, const unsigned int& nu, const bool& with_residuals)
    : CostModelAbstract(model, (unsigned int)0, nu, with_residuals) {}

CostModelSum::CostModelSum(pinocchio::Model* const model, const bool& with_residuals)
    : CostModelAbstract(model, (unsigned int)0, with_residuals) {}

CostModelSum::~CostModelSum() {}

void CostModelSum::addCost(const std::string& name, CostModelAbstract* const cost, const double& weight) {
  std::pair<CostModelContainer::iterator, bool> ret =
      costs_.insert(std::make_pair(name, CostItem(name, cost, weight)));
  if (ret.second == false) {
    std::cout << "Warning: this cost item already existed, we cannot add it" << std::endl;
  } else {
    nr_ += cost->get_nr();
  }
}

void CostModelSum::removeCost(const std::string& name) {
  CostModelContainer::iterator it = costs_.find(name);
  if (it != costs_.end()) {
    nr_ -= it->second.cost->get_nr();
    costs_.erase(it);
  } else {
    std::cout << "Warning: this cost item doesn't exist, we cannot remove it" << std::endl;
  }
}

void CostModelSum::calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u) {
  CostDataSum* d = static_cast<CostDataSum*>(data.get());
  d->cost = 0.;
  unsigned int nr = 0;

  CostModelContainer::iterator it_m, end_m;
  CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = d->costs.begin(), end_d = d->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const CostItem& m_i = it_m->second;
    boost::shared_ptr<CostDataAbstract>& d_i = it_d->second;

    m_i.cost->calc(d_i, x, u);
    d->cost += m_i.weight * d_i->cost;
    if (with_residuals_) {
      const unsigned int& nr_i = m_i.cost->get_nr();
      d->r.segment(nr, nr + nr_i) = sqrt(m_i.weight) * d_i->r;
      nr += nr_i;
    }
  }
}

void CostModelSum::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                            const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                            const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  CostDataSum* d = static_cast<CostDataSum*>(data.get());
  unsigned int nr = 0;
  d->Lx.fill(0);
  d->Lu.fill(0);
  d->Lxx.fill(0);
  d->Lxu.fill(0);
  d->Luu.fill(0);

  CostModelContainer::iterator it_m, end_m;
  CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = d->costs.begin(), end_d = d->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const CostItem& m_i = it_m->second;
    boost::shared_ptr<CostDataAbstract>& d_i = it_d->second;

    m_i.cost->calcDiff(d_i, x, u);
    d->Lx += m_i.weight * d_i->Lx;
    d->Lu += m_i.weight * d_i->Lu;
    d->Lxx += m_i.weight * d_i->Lxx;
    d->Lxu += m_i.weight * d_i->Lxu;
    d->Luu += m_i.weight * d_i->Luu;
    if (with_residuals_) {
      const unsigned int& nr_i = m_i.cost->get_nr();
      d->Rx.block(nr, 0, nr + nr_i, ndx_) = sqrt(m_i.weight) * d_i->Rx;
      d->Ru.block(nr, 0, nr + nr_i, nu_) = sqrt(m_i.weight) * d_i->Ru;
      nr += nr_i;
    }
  }
}

boost::shared_ptr<CostDataAbstract> CostModelSum::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataSum>(this, data);
}

const CostModelSum::CostModelContainer& CostModelSum::get_costs() const { return costs_; }

}  // namespace crocoddyl