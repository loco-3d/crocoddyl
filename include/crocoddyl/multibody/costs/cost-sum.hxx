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

template <typename Scalar>
CostModelSumTpl<Scalar>::CostModelSumTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu)
    : state_(state), nu_(nu), nr_(0), nr_total_(0) {}

template <typename Scalar>
CostModelSumTpl<Scalar>::CostModelSumTpl(boost::shared_ptr<StateMultibody> state)
    : state_(state), nu_(state->get_nv()), nr_(0), nr_total_(0) {}

template <typename Scalar>
CostModelSumTpl<Scalar>::~CostModelSumTpl() {}

template <typename Scalar>
void CostModelSumTpl<Scalar>::addCost(const std::string& name, boost::shared_ptr<CostModelAbstract> cost,
                                      const Scalar& weight, bool active) {
  if (cost->get_nu() != nu_) {
    throw_pretty(name << " cost item doesn't have the same control dimension (it should be " + std::to_string(nu_) +
                             ")");
  }
  std::pair<typename CostModelContainer::iterator, bool> ret =
      costs_.insert(std::make_pair(name, boost::make_shared<CostItem>(name, cost, weight, active)));
  if (ret.second == false) {
    std::cout << "Warning: we couldn't add the " << name << " cost item, it already existed." << std::endl;
  } else if (active) {
    nr_ += cost->get_activation()->get_nr();
    nr_total_ += cost->get_activation()->get_nr();
    std::vector<std::string>::iterator it =
        std::lower_bound(active_.begin(), active_.end(), name, std::greater<std::string>());
    active_.insert(it, name);
  } else if (!active) {
    nr_total_ += cost->get_activation()->get_nr();
    std::vector<std::string>::iterator it =
        std::lower_bound(inactive_.begin(), inactive_.end(), name, std::greater<std::string>());
    inactive_.insert(it, name);
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::removeCost(const std::string& name) {
  typename CostModelContainer::iterator it = costs_.find(name);
  if (it != costs_.end()) {
    nr_ -= it->second->cost->get_activation()->get_nr();
    nr_total_ -= it->second->cost->get_activation()->get_nr();
    costs_.erase(it);
    active_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
    inactive_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
  } else {
    std::cout << "Warning: we couldn't remove the " << name << " cost item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::changeCostStatus(const std::string& name, bool active) {
  typename CostModelContainer::iterator it = costs_.find(name);
  if (it != costs_.end()) {
    if (active && !it->second->active) {
      nr_ += it->second->cost->get_activation()->get_nr();
      std::vector<std::string>::iterator it =
          std::lower_bound(active_.begin(), active_.end(), name, std::greater<std::string>());
      active_.insert(it, name);
      inactive_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
    } else if (!active && it->second->active) {
      nr_ -= it->second->cost->get_activation()->get_nr();
      active_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
      std::vector<std::string>::iterator it =
          std::lower_bound(inactive_.begin(), inactive_.end(), name, std::greater<std::string>());
      inactive_.insert(it, name);
    }
    it->second->active = active;
  } else {
    std::cout << "Warning: we couldn't change the status of the " << name << " cost item, it doesn't exist."
              << std::endl;
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calc(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data,
                                   const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
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

  typename CostModelContainer::iterator it_m, end_m;
  typename CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(), end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<CostItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<CostDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the cost name between model and data ("
                                                    << it_m->first << " != " << it_d->first << ")");

      m_i->cost->calc(d_i, x, u);
      data->cost += m_i->weight * d_i->cost;
    }
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data,
                                       const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
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
  data->Lx.setZero();
  data->Lu.setZero();
  data->Lxx.setZero();
  data->Lxu.setZero();
  data->Luu.setZero();

  typename CostModelContainer::iterator it_m, end_m;
  typename CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(), end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<CostItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<CostDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the cost name between model and data ("
                                                    << it_m->first << " != " << it_d->first << ")");

      m_i->cost->calcDiff(d_i, x, u);
      data->Lx += m_i->weight * d_i->Lx;
      data->Lu += m_i->weight * d_i->Lu;
      data->Lxx += m_i->weight * d_i->Lxx;
      data->Lxu += m_i->weight * d_i->Lxu;
      data->Luu += m_i->weight * d_i->Luu;
    }
  }
}

template <typename Scalar>
boost::shared_ptr<CostDataSumTpl<Scalar> > CostModelSumTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataSumTpl<Scalar> >(this, data);
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calc(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data,
                                   const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data,
                                       const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
const boost::shared_ptr<StateMultibodyTpl<Scalar> >& CostModelSumTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename CostModelSumTpl<Scalar>::CostModelContainer& CostModelSumTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::size_t& CostModelSumTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
const std::size_t& CostModelSumTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
const std::size_t& CostModelSumTpl<Scalar>::get_nr_total() const {
  return nr_total_;
}

template <typename Scalar>
const std::vector<std::string>& CostModelSumTpl<Scalar>::get_active() const {
  return active_;
}

template <typename Scalar>
const std::vector<std::string>& CostModelSumTpl<Scalar>::get_inactive() const {
  return inactive_;
}

template <typename Scalar>
bool CostModelSumTpl<Scalar>::getCostStatus(const std::string& name) const {
  typename CostModelContainer::const_iterator it = costs_.find(name);
  if (it != costs_.end()) {
    return it->second->active;
  } else {
    std::cout << "Warning: we couldn't get the status of the " << name << " cost item, it doesn't exist." << std::endl;
    return false;
  }
}

}  // namespace crocoddyl
