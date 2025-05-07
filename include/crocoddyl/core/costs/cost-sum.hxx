///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
CostModelSumTpl<Scalar>::CostModelSumTpl(std::shared_ptr<StateAbstract> state,
                                         const std::size_t nu)
    : state_(state), nu_(nu), nr_(0), nr_total_(0) {}

template <typename Scalar>
CostModelSumTpl<Scalar>::CostModelSumTpl(std::shared_ptr<StateAbstract> state)
    : state_(state), nu_(state->get_nv()), nr_(0), nr_total_(0) {}

template <typename Scalar>
CostModelSumTpl<Scalar>::~CostModelSumTpl() {}

template <typename Scalar>
void CostModelSumTpl<Scalar>::addCost(const std::string& name,
                                      std::shared_ptr<CostModelAbstract> cost,
                                      const Scalar weight, const bool active) {
  if (cost->get_nu() != nu_) {
    throw_pretty(
        name
        << " cost item doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  std::pair<typename CostModelContainer::iterator, bool> ret =
      costs_.insert(std::make_pair(
          name, std::make_shared<CostItem>(name, cost, weight, active)));
  if (ret.second == false) {
    std::cerr << "Warning: we couldn't add the " << name
              << " cost item, it already existed." << std::endl;
  } else if (active) {
    nr_ += cost->get_activation()->get_nr();
    nr_total_ += cost->get_activation()->get_nr();
    active_set_.insert(name);
  } else if (!active) {
    nr_total_ += cost->get_activation()->get_nr();
    inactive_set_.insert(name);
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::addCost(
    const std::shared_ptr<CostItem>& cost_item) {
  if (cost_item->cost->get_nu() != nu_) {
    throw_pretty(
        cost_item->name
        << " cost item doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  costs_.insert(std::make_pair(cost_item->name, cost_item));
  if (cost_item->active) {
    nr_ += cost_item->cost->get_activation()->get_nr();
    nr_total_ += cost_item->cost->get_activation()->get_nr();
    active_set_.insert(cost_item->name);
  } else {
    nr_total_ += cost_item->cost->get_activation()->get_nr();
    inactive_set_.insert(cost_item->name);
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::removeCost(const std::string& name) {
  typename CostModelContainer::iterator it = costs_.find(name);
  if (it != costs_.end()) {
    nr_ -= it->second->cost->get_activation()->get_nr();
    nr_total_ -= it->second->cost->get_activation()->get_nr();
    costs_.erase(it);
    active_set_.erase(name);
    inactive_set_.erase(name);
  } else {
    std::cerr << "Warning: we couldn't remove the " << name
              << " cost item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::changeCostStatus(const std::string& name,
                                               const bool active) {
  typename CostModelContainer::iterator it = costs_.find(name);
  if (it != costs_.end()) {
    if (active && !it->second->active) {
      nr_ += it->second->cost->get_activation()->get_nr();
      active_set_.insert(name);
      inactive_set_.erase(name);
      it->second->active = active;
    } else if (!active && it->second->active) {
      nr_ -= it->second->cost->get_activation()->get_nr();
      active_set_.erase(name);
      inactive_set_.insert(name);
      it->second->active = active;
    }
  } else {
    std::cerr << "Warning: we couldn't change the status of the " << name
              << " cost item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calc(const std::shared_ptr<CostDataSum>& data,
                                   const Eigen::Ref<const VectorXs>& x,
                                   const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (data->costs.size() != costs_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of cost datas and models");
  }
  data->cost = Scalar(0.);

  typename CostModelContainer::iterator it_m, end_m;
  typename CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(),
      end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<CostItem>& m_i = it_m->second;
    if (m_i->active) {
      const std::shared_ptr<CostDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the cost name between model and data ("
                        << it_m->first << " != " << it_d->first << ")");

      m_i->cost->calc(d_i, x, u);
      data->cost += m_i->weight * d_i->cost;
    }
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calc(const std::shared_ptr<CostDataSum>& data,
                                   const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (data->costs.size() != costs_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of cost datas and models");
  }
  data->cost = Scalar(0.);

  typename CostModelContainer::iterator it_m, end_m;
  typename CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(),
      end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<CostItem>& m_i = it_m->second;
    if (m_i->active) {
      const std::shared_ptr<CostDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the cost name between model and data ("
                        << it_m->first << " != " << it_d->first << ")");

      m_i->cost->calc(d_i, x);
      data->cost += m_i->weight * d_i->cost;
    }
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calcDiff(const std::shared_ptr<CostDataSum>& data,
                                       const Eigen::Ref<const VectorXs>& x,
                                       const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
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
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(),
      end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<CostItem>& m_i = it_m->second;
    if (m_i->active) {
      const std::shared_ptr<CostDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the cost name between model and data ("
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
void CostModelSumTpl<Scalar>::calcDiff(const std::shared_ptr<CostDataSum>& data,
                                       const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (data->costs.size() != costs_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of cost datas and models");
  }
  data->Lx.setZero();
  data->Lxx.setZero();

  typename CostModelContainer::iterator it_m, end_m;
  typename CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(),
      end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<CostItem>& m_i = it_m->second;
    if (m_i->active) {
      const std::shared_ptr<CostDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the cost name between model and data ("
                        << it_m->first << " != " << it_d->first << ")");

      m_i->cost->calcDiff(d_i, x);
      data->Lx += m_i->weight * d_i->Lx;
      data->Lxx += m_i->weight * d_i->Lxx;
    }
  }
}

template <typename Scalar>
std::shared_ptr<CostDataSumTpl<Scalar> > CostModelSumTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<CostDataSum>(
      Eigen::aligned_allocator<CostDataSum>(), this, data);
}

template <typename Scalar>
template <typename NewScalar>
CostModelSumTpl<NewScalar> CostModelSumTpl<Scalar>::cast() const {
  typedef CostModelSumTpl<NewScalar> ReturnType;
  typedef CostItemTpl<NewScalar> CostType;
  ReturnType ret(state_->template cast<NewScalar>(), nu_);
  typename CostModelContainer::const_iterator it_m, end_m;
  for (it_m = costs_.begin(), end_m = costs_.end(); it_m != end_m; ++it_m) {
    const std::string name = it_m->first;
    const CostType& m_i = it_m->second->template cast<NewScalar>();
    ret.addCost(name, m_i.cost, m_i.weight, m_i.active);
  }
  return ret;
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar> >&
CostModelSumTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename CostModelSumTpl<Scalar>::CostModelContainer&
CostModelSumTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
std::size_t CostModelSumTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t CostModelSumTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
std::size_t CostModelSumTpl<Scalar>::get_nr_total() const {
  return nr_total_;
}

template <typename Scalar>
const std::set<std::string>& CostModelSumTpl<Scalar>::get_active_set() const {
  return active_set_;
}

template <typename Scalar>
const std::set<std::string>& CostModelSumTpl<Scalar>::get_inactive_set() const {
  return inactive_set_;
}

template <typename Scalar>
bool CostModelSumTpl<Scalar>::getCostStatus(const std::string& name) const {
  typename CostModelContainer::const_iterator it = costs_.find(name);
  if (it != costs_.end()) {
    return it->second->active;
  } else {
    std::cerr << "Warning: we couldn't get the status of the " << name
              << " cost item, it doesn't exist." << std::endl;
    return false;
  }
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const CostModelSumTpl<Scalar>& model) {
  const std::set<std::string>& active = model.get_active_set();
  const std::set<std::string>& inactive = model.get_inactive_set();
  os << "CostModelSum:" << std::endl;
  os << "  Active:" << std::endl;
  for (std::set<std::string>::const_iterator it = active.begin();
       it != active.end(); ++it) {
    const std::shared_ptr<typename CostModelSumTpl<Scalar>::CostItem>&
        cost_item = model.get_costs().find(*it)->second;
    if (it != --active.end()) {
      os << "    " << *it << ": " << *cost_item << std::endl;
    } else {
      os << "    " << *it << ": " << *cost_item << std::endl;
    }
  }
  os << "  Inactive:" << std::endl;
  for (std::set<std::string>::const_iterator it = inactive.begin();
       it != inactive.end(); ++it) {
    const std::shared_ptr<typename CostModelSumTpl<Scalar>::CostItem>&
        cost_item = model.get_costs().find(*it)->second;
    if (it != --inactive.end()) {
      os << "    " << *it << ": " << *cost_item << std::endl;
    } else {
      os << "    " << *it << ": " << *cost_item;
    }
  }
  return os;
}

}  // namespace crocoddyl
