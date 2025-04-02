///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ConstraintModelManagerTpl<Scalar>::ConstraintModelManagerTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nu)
    : state_(state),
      lb_(0),
      ub_(0),
      nu_(nu),
      ng_(0),
      nh_(0),
      ng_T_(0),
      nh_T_(0) {}

template <typename Scalar>
ConstraintModelManagerTpl<Scalar>::ConstraintModelManagerTpl(
    std::shared_ptr<StateAbstract> state)
    : state_(state),
      lb_(0),
      ub_(0),
      nu_(state->get_nv()),
      ng_(0),
      nh_(0),
      ng_T_(0),
      nh_T_(0) {}

template <typename Scalar>
ConstraintModelManagerTpl<Scalar>::~ConstraintModelManagerTpl() {}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::addConstraint(
    const std::string& name,
    std::shared_ptr<ConstraintModelAbstract> constraint, const bool active) {
  if (constraint->get_nu() != nu_) {
    throw_pretty(name << " constraint item doesn't have the same control "
                         "dimension (it should be " +
                             std::to_string(nu_) + ")");
  }
  std::pair<typename ConstraintModelContainer::iterator, bool> ret =
      constraints_.insert(std::make_pair(
          name, std::make_shared<ConstraintItem>(name, constraint, active)));
  if (ret.second == false) {
    std::cout << "Warning: we couldn't add the " << name
              << " constraint item, it already existed." << std::endl;
  } else if (active) {
    ng_ += constraint->get_ng();
    nh_ += constraint->get_nh();
    if (constraint->get_T_constraint()) {
      ng_T_ += constraint->get_ng();
      nh_T_ += constraint->get_nh();
    }
    active_set_.insert(name);
    lb_.resize(ng_);
    ub_.resize(ng_);
  } else if (!active) {
    inactive_set_.insert(name);
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::removeConstraint(
    const std::string& name) {
  typename ConstraintModelContainer::iterator it = constraints_.find(name);
  if (it != constraints_.end()) {
    if (it->second->active) {
      ng_ -= it->second->constraint->get_ng();
      nh_ -= it->second->constraint->get_nh();
      if (it->second->constraint->get_T_constraint()) {
        ng_T_ -= it->second->constraint->get_ng();
        nh_T_ -= it->second->constraint->get_nh();
      }
      lb_.resize(ng_);
      ub_.resize(ng_);
      active_set_.erase(name);
    } else {
      inactive_set_.erase(name);
    }
    constraints_.erase(it);
  } else {
    std::cout << "Warning: we couldn't remove the " << name
              << " constraint item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::changeConstraintStatus(
    const std::string& name, bool active) {
  typename ConstraintModelContainer::iterator it = constraints_.find(name);
  if (it != constraints_.end()) {
    if (active && !it->second->active) {
      ng_ += it->second->constraint->get_ng();
      nh_ += it->second->constraint->get_nh();
      if (it->second->constraint->get_T_constraint()) {
        ng_T_ += it->second->constraint->get_ng();
        nh_T_ += it->second->constraint->get_nh();
      }
      active_set_.insert(name);
      inactive_set_.erase(name);
      it->second->active = active;
      lb_.resize(ng_);
      ub_.resize(ng_);
    } else if (!active && it->second->active) {
      ng_ -= it->second->constraint->get_ng();
      nh_ -= it->second->constraint->get_nh();
      if (it->second->constraint->get_T_constraint()) {
        ng_T_ -= it->second->constraint->get_ng();
        nh_T_ -= it->second->constraint->get_nh();
      }
      active_set_.erase(name);
      inactive_set_.insert(name);
      it->second->active = active;
      lb_.resize(ng_);
      ub_.resize(ng_);
    }
  } else {
    std::cout << "Warning: we couldn't change the status of the " << name
              << " constraint item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::calc(
    const std::shared_ptr<ConstraintDataManager>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
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
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty(
        "Invalid argument: "
        << "it doesn't match the number of constraint datas and models");
  }
  assert_pretty(static_cast<std::size_t>(data->g.size()) == ng_,
                "the dimension of data.g doesn't correspond with ng=" << ng_);
  assert_pretty(static_cast<std::size_t>(data->h.size()) == nh_,
                "the dimension of data.h doesn't correspond with nh=" << nh_);
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;

  typename ConstraintModelContainer::iterator it_m, end_m;
  typename ConstraintDataContainer::iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(),
      it_d = data->constraints.begin(), end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ConstraintItem>& m_i = it_m->second;
    if (m_i->active) {
      const std::shared_ptr<ConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(
          it_m->first == it_d->first,
          "it doesn't match the constraint name between model and data ("
              << it_m->first << " != " << it_d->first << ")");

      m_i->constraint->calc(d_i, x, u);
      const std::size_t ng = m_i->constraint->get_ng();
      const std::size_t nh = m_i->constraint->get_nh();
      data->g.segment(ng_i, ng) = d_i->g;
      data->h.segment(nh_i, nh) = d_i->h;
      lb_.segment(ng_i, ng) = m_i->constraint->get_lb();
      ub_.segment(ng_i, ng) = m_i->constraint->get_ub();
      ng_i += ng;
      nh_i += nh;
    }
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::calc(
    const std::shared_ptr<ConstraintDataManager>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty(
        "Invalid argument: "
        << "it doesn't match the number of constraint datas and models");
  }
  assert_pretty(static_cast<std::size_t>(data->g.size()) == ng_T_,
                "the dimension of data.g doesn't correspond with ng=" << ng_T_);
  assert_pretty(static_cast<std::size_t>(data->h.size()) == nh_T_,
                "the dimension of data.h doesn't correspond with nh=" << nh_T_);
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;

  typename ConstraintModelContainer::iterator it_m, end_m;
  typename ConstraintDataContainer::iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(),
      it_d = data->constraints.begin(), end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ConstraintItem>& m_i = it_m->second;
    if (m_i->active && m_i->constraint->get_T_constraint()) {
      const std::shared_ptr<ConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(
          it_m->first == it_d->first,
          "it doesn't match the constraint name between model and data ("
              << it_m->first << " != " << it_d->first << ")");

      m_i->constraint->calc(d_i, x);
      const std::size_t ng = m_i->constraint->get_ng();
      const std::size_t nh = m_i->constraint->get_nh();
      data->g.segment(ng_i, ng) = d_i->g;
      data->h.segment(nh_i, nh) = d_i->h;
      lb_.segment(ng_i, ng) = m_i->constraint->get_lb();
      ub_.segment(ng_i, ng) = m_i->constraint->get_ub();
      ng_i += ng;
      nh_i += nh;
    }
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::calcDiff(
    const std::shared_ptr<ConstraintDataManager>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
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
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty(
        "Invalid argument: "
        << "it doesn't match the number of constraint datas and models");
  }
  assert_pretty(static_cast<std::size_t>(data->Gx.rows()) == ng_,
                "the dimension of data.Gx doesn't correspond with ng=" << ng_);
  assert_pretty(static_cast<std::size_t>(data->Gu.rows()) == ng_,
                "the dimension of data.Gu doesn't correspond with ng=" << ng_);
  assert_pretty(
      static_cast<std::size_t>(data->Hx.rows()) == nh_,
      "the dimension of data.Hx,u doesn't correspond with nh=" << nh_);
  const std::size_t ndx = state_->get_ndx();
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;

  typename ConstraintModelContainer::iterator it_m, end_m;
  typename ConstraintDataContainer::iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(),
      it_d = data->constraints.begin(), end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ConstraintItem>& m_i = it_m->second;
    if (m_i->active) {
      const std::shared_ptr<ConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(
          it_m->first == it_d->first,
          "it doesn't match the constraint name between model and data ("
              << it_m->first << " != " << it_d->first << ")");

      m_i->constraint->calcDiff(d_i, x, u);
      const std::size_t ng = m_i->constraint->get_ng();
      const std::size_t nh = m_i->constraint->get_nh();
      data->Gx.block(ng_i, 0, ng, ndx) = d_i->Gx;
      data->Gu.block(ng_i, 0, ng, nu_) = d_i->Gu;
      data->Hx.block(nh_i, 0, nh, ndx) = d_i->Hx;
      data->Hu.block(nh_i, 0, nh, nu_) = d_i->Hu;
      ng_i += ng;
      nh_i += nh;
    }
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::calcDiff(
    const std::shared_ptr<ConstraintDataManager>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty(
        "Invalid argument: "
        << "it doesn't match the number of constraint datas and models");
  }
  assert_pretty(
      static_cast<std::size_t>(data->Gx.rows()) == ng_T_,
      "the dimension of data.Gx,u doesn't correspond with ng=" << ng_T_);
  assert_pretty(
      static_cast<std::size_t>(data->Hx.rows()) == nh_T_,
      "the dimension of data.Hx,u doesn't correspond with nh=" << nh_T_);
  const std::size_t ndx = state_->get_ndx();
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;

  typename ConstraintModelContainer::iterator it_m, end_m;
  typename ConstraintDataContainer::iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(),
      it_d = data->constraints.begin(), end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ConstraintItem>& m_i = it_m->second;
    if (m_i->active && m_i->constraint->get_T_constraint()) {
      const std::shared_ptr<ConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(
          it_m->first == it_d->first,
          "it doesn't match the constraint name between model and data ("
              << it_m->first << " != " << it_d->first << ")");

      m_i->constraint->calcDiff(d_i, x);
      const std::size_t ng = m_i->constraint->get_ng();
      const std::size_t nh = m_i->constraint->get_nh();
      data->Gx.block(ng_i, 0, ng, ndx) = d_i->Gx;
      data->Hx.block(nh_i, 0, nh, ndx) = d_i->Hx;
      ng_i += ng;
      nh_i += nh;
    }
  }
}

template <typename Scalar>
std::shared_ptr<ConstraintDataManagerTpl<Scalar> >
ConstraintModelManagerTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<ConstraintDataManager>(
      Eigen::aligned_allocator<ConstraintDataManager>(), this, data);
}

template <typename Scalar>
template <typename NewScalar>
ConstraintModelManagerTpl<NewScalar> ConstraintModelManagerTpl<Scalar>::cast()
    const {
  typedef ConstraintModelManagerTpl<NewScalar> ReturnType;
  typedef ConstraintItemTpl<NewScalar> ConstraintType;
  ReturnType ret(state_->template cast<NewScalar>(), nu_);
  typename ConstraintModelContainer::const_iterator it_m, end_m;
  for (it_m = constraints_.begin(), end_m = constraints_.end(); it_m != end_m;
       ++it_m) {
    const std::string name = it_m->first;
    const ConstraintType& m_i = it_m->second->template cast<NewScalar>();
    ret.addConstraint(name, m_i.constraint, m_i.active);
  }
  return ret;
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar> >&
ConstraintModelManagerTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename ConstraintModelManagerTpl<Scalar>::ConstraintModelContainer&
ConstraintModelManagerTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
std::size_t ConstraintModelManagerTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t ConstraintModelManagerTpl<Scalar>::get_ng() const {
  return ng_;
}

template <typename Scalar>
std::size_t ConstraintModelManagerTpl<Scalar>::get_nh() const {
  return nh_;
}

template <typename Scalar>
std::size_t ConstraintModelManagerTpl<Scalar>::get_ng_T() const {
  return ng_T_;
}

template <typename Scalar>
std::size_t ConstraintModelManagerTpl<Scalar>::get_nh_T() const {
  return nh_T_;
}

template <typename Scalar>
const std::set<std::string>& ConstraintModelManagerTpl<Scalar>::get_active_set()
    const {
  return active_set_;
}

template <typename Scalar>
const std::set<std::string>&
ConstraintModelManagerTpl<Scalar>::get_inactive_set() const {
  return inactive_set_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ConstraintModelManagerTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ConstraintModelManagerTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
bool ConstraintModelManagerTpl<Scalar>::getConstraintStatus(
    const std::string& name) const {
  typename ConstraintModelContainer::const_iterator it =
      constraints_.find(name);
  if (it != constraints_.end()) {
    return it->second->active;
  } else {
    std::cout << "Warning: we couldn't get the status of the " << name
              << " constraint item, it doesn't exist." << std::endl;
    return false;
  }
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ConstraintModelManagerTpl<Scalar>& model) {
  const std::set<std::string>& active = model.get_active_set();
  const std::set<std::string>& inactive = model.get_inactive_set();
  os << "ConstraintModelManagerTpl:" << std::endl;
  os << "  Active:" << std::endl;
  for (std::set<std::string>::const_iterator it = active.begin();
       it != active.end(); ++it) {
    const std::shared_ptr<
        typename ConstraintModelManagerTpl<Scalar>::ConstraintItem>&
        constraint_item = model.get_constraints().find(*it)->second;
    if (it != --active.end()) {
      os << "    " << *it << ": " << *constraint_item << std::endl;
    } else {
      os << "    " << *it << ": " << *constraint_item << std::endl;
    }
  }
  os << "  Inactive:" << std::endl;
  for (std::set<std::string>::const_iterator it = inactive.begin();
       it != inactive.end(); ++it) {
    const std::shared_ptr<
        typename ConstraintModelManagerTpl<Scalar>::ConstraintItem>&
        constraint_item = model.get_constraints().find(*it)->second;
    if (it != --inactive.end()) {
      os << "    " << *it << ": " << *constraint_item << std::endl;
    } else {
      os << "    " << *it << ": " << *constraint_item;
    }
  }
  return os;
}

}  // namespace crocoddyl
