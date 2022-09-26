///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2022, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ConstraintModelManagerTpl<Scalar>::ConstraintModelManagerTpl(boost::shared_ptr<StateAbstract> state,
                                                             const std::size_t nu)
    : state_(state),
      nu_(nu),
      ng_internal_(0),
      nh_internal_(0),
      ng_(&ng_internal_),
      ng_total_(0),
      nh_(&nh_internal_),
      nh_total_(0) {}

template <typename Scalar>
ConstraintModelManagerTpl<Scalar>::ConstraintModelManagerTpl(boost::shared_ptr<StateAbstract> state)
    : state_(state),
      nu_(state->get_nv()),
      ng_internal_(0),
      nh_internal_(0),
      ng_(&ng_internal_),
      ng_total_(0),
      nh_(&nh_internal_),
      nh_total_(0) {}

template <typename Scalar>
ConstraintModelManagerTpl<Scalar>::~ConstraintModelManagerTpl() {}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::addConstraint(const std::string& name,
                                                      boost::shared_ptr<ConstraintModelAbstract> constraint,
                                                      const bool active) {
  if (constraint->get_nu() != nu_) {
    throw_pretty(name << " constraint item doesn't have the same control dimension (it should be " +
                             std::to_string(nu_) + ")");
  }
  std::pair<typename ConstraintModelContainer::iterator, bool> ret =
      constraints_.insert(std::make_pair(name, boost::make_shared<ConstraintItem>(name, constraint, active)));
  if (ret.second == false) {
    std::cout << "Warning: we couldn't add the " << name << " constraint item, it already existed." << std::endl;
  } else if (active) {
    *ng_ += constraint->get_ng();
    *nh_ += constraint->get_nh();
    ng_total_ += constraint->get_ng();
    nh_total_ += constraint->get_nh();
    std::vector<std::string>::iterator it =
        std::lower_bound(active_.begin(), active_.end(), name, std::less<std::string>());
    active_.insert(it, name);
  } else if (!active) {
    ng_total_ += constraint->get_ng();
    nh_total_ += constraint->get_nh();
    std::vector<std::string>::iterator it =
        std::lower_bound(inactive_.begin(), inactive_.end(), name, std::less<std::string>());
    inactive_.insert(it, name);
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::removeConstraint(const std::string& name) {
  typename ConstraintModelContainer::iterator it = constraints_.find(name);
  if (it != constraints_.end()) {
    *ng_ -= it->second->constraint->get_ng();
    *nh_ -= it->second->constraint->get_nh();
    ng_total_ -= it->second->constraint->get_ng();
    nh_total_ -= it->second->constraint->get_nh();
    constraints_.erase(it);
    active_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
    inactive_.erase(std::remove(inactive_.begin(), inactive_.end(), name), inactive_.end());
  } else {
    std::cout << "Warning: we couldn't remove the " << name << " constraint item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::changeConstraintStatus(const std::string& name, bool active) {
  typename ConstraintModelContainer::iterator it = constraints_.find(name);
  if (it != constraints_.end()) {
    if (active && !it->second->active) {
      *ng_ += it->second->constraint->get_ng();
      *nh_ += it->second->constraint->get_nh();
      std::vector<std::string>::iterator it =
          std::lower_bound(active_.begin(), active_.end(), name, std::less<std::string>());
      active_.insert(it, name);
      inactive_.erase(std::remove(inactive_.begin(), inactive_.end(), name), inactive_.end());
    } else if (!active && it->second->active) {
      *ng_ -= it->second->constraint->get_ng();
      *nh_ -= it->second->constraint->get_nh();
      active_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
      std::vector<std::string>::iterator it =
          std::lower_bound(inactive_.begin(), inactive_.end(), name, std::less<std::string>());
      inactive_.insert(it, name);
    }
    it->second->active = active;
  } else {
    std::cout << "Warning: we couldn't change the status of the " << name << " constraint item, it doesn't exist."
              << std::endl;
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataManager>& data,
                                             const Eigen::Ref<const VectorXs>& x,
                                             const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and models");
  }
  assert_pretty(static_cast<std::size_t>(data->g.size()) == *ng_,
                "the dimension of data.g doesn't correspond with ng=" << *ng_);
  assert_pretty(static_cast<std::size_t>(data->h.size()) == *nh_,
                "the dimension of data.h doesn't correspond with nh=" << *nh_);
  data->g.setZero();
  data->h.setZero();
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;

  typename ConstraintModelContainer::iterator it_m, end_m;
  typename ConstraintDataContainer::iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(), it_d = data->constraints.begin(),
      end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ConstraintItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<ConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the constraint name between model and data ("
                                                    << it_m->first << " != " << it_d->first << ")");

      m_i->constraint->calc(d_i, x, u);
      const std::size_t ng = m_i->constraint->get_ng();
      const std::size_t nh = m_i->constraint->get_nh();
      data->g.segment(ng_i, ng) = d_i->g;
      data->h.segment(nh_i, nh) = d_i->h;
      ng_i += ng;
      nh_i += nh;
    }
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataManager>& data,
                                             const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and models");
  }
  assert_pretty(static_cast<std::size_t>(data->g.size()) == *ng_,
                "the dimension of data.g doesn't correspond with ng=" << *ng_);
  assert_pretty(static_cast<std::size_t>(data->h.size()) == *nh_,
                "the dimension of data.h doesn't correspond with nh=" << *nh_);
  data->g.setZero();
  data->h.setZero();
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;

  typename ConstraintModelContainer::iterator it_m, end_m;
  typename ConstraintDataContainer::iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(), it_d = data->constraints.begin(),
      end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ConstraintItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<ConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the constraint name between model and data ("
                                                    << it_m->first << " != " << it_d->first << ")");

      m_i->constraint->calc(d_i, x);
      const std::size_t ng = m_i->constraint->get_ng();
      const std::size_t nh = m_i->constraint->get_nh();
      data->g.segment(ng_i, ng) = d_i->g;
      data->h.segment(nh_i, nh) = d_i->h;
      ng_i += ng;
      nh_i += nh;
    }
  }
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataManager>& data,
                                                 const Eigen::Ref<const VectorXs>& x,
                                                 const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and models");
  }
  assert_pretty(static_cast<std::size_t>(data->Gx.rows()) == *ng_,
                "the dimension of data.Gx,u doesn't correspond with ng=" << *ng_);
  assert_pretty(static_cast<std::size_t>(data->Hx.rows()) == *nh_,
                "the dimension of data.Hx,u doesn't correspond with nh=" << *nh_);
  const std::size_t ndx = state_->get_ndx();
  data->Gx.setZero();
  data->Gu.setZero();
  data->Hx.setZero();
  data->Hu.setZero();
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;

  typename ConstraintModelContainer::iterator it_m, end_m;
  typename ConstraintDataContainer::iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(), it_d = data->constraints.begin(),
      end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ConstraintItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<ConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the constraint name between model and data ("
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
void ConstraintModelManagerTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataManager>& data,
                                                 const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (data->constraints.size() != constraints_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of constraint datas and models");
  }
  assert_pretty(static_cast<std::size_t>(data->Gx.rows()) == *ng_,
                "the dimension of data.Gx,u doesn't correspond with ng=" << *ng_);
  assert_pretty(static_cast<std::size_t>(data->Hx.rows()) == *nh_,
                "the dimension of data.Hx,u doesn't correspond with nh=" << *nh_);
  const std::size_t ndx = state_->get_ndx();
  data->Gx.setZero();
  data->Gu.setZero();
  data->Hx.setZero();
  data->Hu.setZero();
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;

  typename ConstraintModelContainer::iterator it_m, end_m;
  typename ConstraintDataContainer::iterator it_d, end_d;
  for (it_m = constraints_.begin(), end_m = constraints_.end(), it_d = data->constraints.begin(),
      end_d = data->constraints.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ConstraintItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<ConstraintDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the constraint name between model and data ("
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
boost::shared_ptr<ConstraintDataManagerTpl<Scalar> > ConstraintModelManagerTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<ConstraintDataManager>(Eigen::aligned_allocator<ConstraintDataManager>(), this, data);
}

template <typename Scalar>
const boost::shared_ptr<StateAbstractTpl<Scalar> >& ConstraintModelManagerTpl<Scalar>::get_state() const {
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
  return *ng_;
}

template <typename Scalar>
std::size_t ConstraintModelManagerTpl<Scalar>::get_ng_total() const {
  return ng_total_;
}

template <typename Scalar>
std::size_t ConstraintModelManagerTpl<Scalar>::get_nh() const {
  return *nh_;
}

template <typename Scalar>
std::size_t ConstraintModelManagerTpl<Scalar>::get_nh_total() const {
  return nh_total_;
}

template <typename Scalar>
const std::vector<std::string>& ConstraintModelManagerTpl<Scalar>::get_active() const {
  return active_;
}

template <typename Scalar>
const std::vector<std::string>& ConstraintModelManagerTpl<Scalar>::get_inactive() const {
  return inactive_;
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::shareDimensions(ActionModelAbstractTpl<Scalar>* const model) {
  ng_ = model->ng_;
  nh_ = model->nh_;
}

template <typename Scalar>
void ConstraintModelManagerTpl<Scalar>::shareDimensions(DifferentialActionModelAbstractTpl<Scalar>* const model) {
  ng_ = model->ng_;
  nh_ = model->nh_;
}

template <typename Scalar>
bool ConstraintModelManagerTpl<Scalar>::getConstraintStatus(const std::string& name) const {
  typename ConstraintModelContainer::const_iterator it = constraints_.find(name);
  if (it != constraints_.end()) {
    return it->second->active;
  } else {
    std::cout << "Warning: we couldn't get the status of the " << name << " constraint item, it doesn't exist."
              << std::endl;
    return false;
  }
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const ConstraintModelManagerTpl<Scalar>& model) {
  const std::vector<std::string>& active = model.get_active();
  const std::vector<std::string>& inactive = model.get_inactive();
  os << "ConstraintModelManagerTpl:" << std::endl;
  os << "  Active:" << std::endl;
  for (std::vector<std::string>::const_iterator it = active.begin(); it != active.end(); ++it) {
    const boost::shared_ptr<typename ConstraintModelManagerTpl<Scalar>::ConstraintItem>& constraint_item =
        model.get_constraints().find(*it)->second;
    if (it != --active.end()) {
      os << "    " << *it << ": " << *constraint_item << std::endl;
    } else {
      os << "    " << *it << ": " << *constraint_item << std::endl;
    }
  }
  os << "  Inactive:" << std::endl;
  for (std::vector<std::string>::const_iterator it = inactive.begin(); it != inactive.end(); ++it) {
    const boost::shared_ptr<typename ConstraintModelManagerTpl<Scalar>::ConstraintItem>& constraint_item =
        model.get_constraints().find(*it)->second;
    if (it != --inactive.end()) {
      os << "    " << *it << ": " << *constraint_item << std::endl;
    } else {
      os << "    " << *it << ": " << *constraint_item;
    }
  }
  return os;
}

}  // namespace crocoddyl
