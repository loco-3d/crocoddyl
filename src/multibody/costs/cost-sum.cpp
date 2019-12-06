///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"

namespace crocoddyl {

CostModelSum::CostModelSum(boost::shared_ptr<StateMultibody> state, const std::size_t& nu, const bool& with_residuals)
    : state_(state), nu_(nu), nr_(0), with_residuals_(with_residuals) {}

CostModelSum::CostModelSum(boost::shared_ptr<StateMultibody> state, const bool& with_residuals)
    : state_(state), nu_(state->get_nv()), nr_(0), with_residuals_(with_residuals) {}

CostModelSum::~CostModelSum() {}

void CostModelSum::addCost(const std::string& name, boost::shared_ptr<CostModelAbstract> cost, const double& weight) {
  assert_pretty(cost->get_nu() == nu_, "Cost item doesn't have the same control dimension");
  std::pair<CostModelContainer::iterator, bool> ret =
      costs_.insert(std::make_pair(name, CostItem(name, cost, weight)));
  if (ret.second == false) {
    std::cout << "Warning: this cost item already existed, we cannot add it" << std::endl;
  } else {
    nr_ += cost->get_activation()->get_nr();
  }
}

void CostModelSum::removeCost(const std::string& name) {
  CostModelContainer::iterator it = costs_.find(name);
  if (it != costs_.end()) {
    nr_ -= it->second.cost->get_activation()->get_nr();
    costs_.erase(it);
  } else {
    std::cout << "Warning: this cost item doesn't exist, we cannot remove it" << std::endl;
  }
}

void CostModelSum::calc(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (data->costs.size() != costs_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of cost datas and models");
  }
  data->cost = 0.;
  std::size_t nr = 0;

  CostModelContainer::iterator it_m, end_m;
  CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(), end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const CostItem& m_i = it_m->second;
    boost::shared_ptr<CostDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first, "it doesn't match the cost name between data and model");

    m_i.cost->calc(d_i, x, u);
    data->cost += m_i.weight * d_i->cost;
    if (with_residuals_) {
      const std::size_t& nr_i = m_i.cost->get_activation()->get_nr();
      data->r.segment(nr, nr_i) = sqrt(m_i.weight) * d_i->r;
      nr += nr_i;
    }
  }
}

void CostModelSum::calcDiff(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                            const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (data->costs.size() != costs_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of cost datas and models");
  }
  if (recalc) {
    calc(data, x, u);
  }
  std::size_t nr = 0;
  data->Lx.fill(0);
  data->Lu.fill(0);
  data->Lxx.fill(0);
  data->Lxu.fill(0);
  data->Luu.fill(0);

  const std::size_t& ndx = state_->get_ndx();
  CostModelContainer::iterator it_m, end_m;
  CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(), end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const CostItem& m_i = it_m->second;
    boost::shared_ptr<CostDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first, "it doesn't match the cost name between data and model");

    m_i.cost->calcDiff(d_i, x, u);
    data->Lx += m_i.weight * d_i->Lx;
    data->Lu += m_i.weight * d_i->Lu;
    data->Lxx += m_i.weight * d_i->Lxx;
    data->Lxu += m_i.weight * d_i->Lxu;
    data->Luu += m_i.weight * d_i->Luu;
    if (with_residuals_) {
      const std::size_t& nr_i = m_i.cost->get_activation()->get_nr();
      data->Rx.block(nr, 0, nr_i, ndx) = sqrt(m_i.weight) * d_i->Rx;
      data->Ru.block(nr, 0, nr_i, nu_) = sqrt(m_i.weight) * d_i->Ru;
      nr += nr_i;
    }
  }
}

boost::shared_ptr<CostDataSum> CostModelSum::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataSum>(this, data);
}

void CostModelSum::calc(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void CostModelSum::calcDiff(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

const boost::shared_ptr<StateMultibody>& CostModelSum::get_state() const { return state_; }

const CostModelSum::CostModelContainer& CostModelSum::get_costs() const { return costs_; }

const std::size_t& CostModelSum::get_nu() const { return nu_; }

const std::size_t& CostModelSum::get_nr() const { return nr_; }

}  // namespace crocoddyl
