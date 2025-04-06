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
ContactModelMultipleTpl<Scalar>::ContactModelMultipleTpl(
    std::shared_ptr<StateMultibody> state, const std::size_t nu)
    : state_(state),
      nc_(0),
      nc_total_(0),
      nu_(nu),
      compute_all_contacts_(false) {}

template <typename Scalar>
ContactModelMultipleTpl<Scalar>::ContactModelMultipleTpl(
    std::shared_ptr<StateMultibody> state)
    : state_(state),
      nc_(0),
      nc_total_(0),
      nu_(state->get_nv()),
      compute_all_contacts_(false) {}

template <typename Scalar>
ContactModelMultipleTpl<Scalar>::~ContactModelMultipleTpl() {}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::addContact(
    const std::string& name, std::shared_ptr<ContactModelAbstract> contact,
    const bool active) {
  if (contact->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << name
                 << " contact item doesn't have the same control dimension (" +
                        std::to_string(nu_) + ")");
  }
  std::pair<typename ContactModelContainer::iterator, bool> ret =
      contacts_.insert(std::make_pair(
          name, std::make_shared<ContactItem>(name, contact, active)));
  if (ret.second == false) {
    std::cerr << "Warning: we couldn't add the " << name
              << " contact item, it already existed." << std::endl;
  } else if (active) {
    nc_ += contact->get_nc();
    nc_total_ += contact->get_nc();
    active_set_.insert(name);
  } else if (!active) {
    nc_total_ += contact->get_nc();
    inactive_set_.insert(name);
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::removeContact(const std::string& name) {
  typename ContactModelContainer::iterator it = contacts_.find(name);
  if (it != contacts_.end()) {
    nc_ -= it->second->contact->get_nc();
    nc_total_ -= it->second->contact->get_nc();
    contacts_.erase(it);
    active_set_.erase(name);
    inactive_set_.erase(name);
  } else {
    std::cerr << "Warning: we couldn't remove the " << name
              << " contact item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::changeContactStatus(
    const std::string& name, const bool active) {
  typename ContactModelContainer::iterator it = contacts_.find(name);
  if (it != contacts_.end()) {
    if (active && !it->second->active) {
      nc_ += it->second->contact->get_nc();
      active_set_.insert(name);
      inactive_set_.erase(name);
    } else if (!active && it->second->active) {
      nc_ -= it->second->contact->get_nc();
      inactive_set_.insert(name);
      active_set_.erase(name);
    }
    // "else" case: Contact status unchanged - already in desired state
    it->second->active = active;
  } else {
    std::cerr << "Warning: we couldn't change the status of the " << name
              << " contact item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::calc(
    const std::shared_ptr<ContactDataMultiple>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (data->contacts.size() != contacts_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of contact datas and models");
  }

  std::size_t nc = 0;
  const std::size_t nv = state_->get_nv();
  typename ContactModelContainer::iterator it_m, end_m;
  typename ContactDataContainer::iterator it_d, end_d;
  if (compute_all_contacts_) {
    for (it_m = contacts_.begin(), end_m = contacts_.end(),
        it_d = data->contacts.begin(), end_d = data->contacts.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      const std::size_t nc_i = m_i->contact->get_nc();
      if (m_i->active) {
        const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
        assert_pretty(
            it_m->first == it_d->first,
            "it doesn't match the contact name between model and data ("
                << it_m->first << " != " << it_d->first << ")");
        m_i->contact->calc(d_i, x);
        data->a0.segment(nc, nc_i) = d_i->a0;
        data->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
      } else {
        data->a0.segment(nc, nc_i).setZero();
        data->Jc.block(nc, 0, nc_i, nv).setZero();
      }
      nc += nc_i;
    }
  } else {
    for (it_m = contacts_.begin(), end_m = contacts_.end(),
        it_d = data->contacts.begin(), end_d = data->contacts.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      if (m_i->active) {
        const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
        assert_pretty(
            it_m->first == it_d->first,
            "it doesn't match the contact name between model and data ("
                << it_m->first << " != " << it_d->first << ")");

        m_i->contact->calc(d_i, x);
        const std::size_t nc_i = m_i->contact->get_nc();
        data->a0.segment(nc, nc_i) = d_i->a0;
        data->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
        nc += nc_i;
      }
    }
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::calcDiff(
    const std::shared_ptr<ContactDataMultiple>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (data->contacts.size() != contacts_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of contact datas and models");
  }

  std::size_t nc = 0;
  const std::size_t ndx = state_->get_ndx();
  typename ContactModelContainer::iterator it_m, end_m;
  typename ContactDataContainer::iterator it_d, end_d;
  if (compute_all_contacts_) {
    for (it_m = contacts_.begin(), end_m = contacts_.end(),
        it_d = data->contacts.begin(), end_d = data->contacts.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      const std::size_t nc_i = m_i->contact->get_nc();
      if (m_i->active) {
        assert_pretty(
            it_m->first == it_d->first,
            "it doesn't match the contact name between model and data ("
                << it_m->first << " != " << it_d->first << ")");
        const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;

        m_i->contact->calcDiff(d_i, x);
        data->da0_dx.block(nc, 0, nc_i, ndx) = d_i->da0_dx;
      } else {
        data->da0_dx.block(nc, 0, nc_i, ndx).setZero();
      }
      nc += nc_i;
    }
  } else {
    for (it_m = contacts_.begin(), end_m = contacts_.end(),
        it_d = data->contacts.begin(), end_d = data->contacts.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      if (m_i->active) {
        const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
        assert_pretty(
            it_m->first == it_d->first,
            "it doesn't match the contact name between model and data ("
                << it_m->first << " != " << it_d->first << ")");

        m_i->contact->calcDiff(d_i, x);
        const std::size_t nc_i = m_i->contact->get_nc();
        data->da0_dx.block(nc, 0, nc_i, ndx) = d_i->da0_dx;
        nc += nc_i;
      }
    }
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateAcceleration(
    const std::shared_ptr<ContactDataMultiple>& data,
    const VectorXs& dv) const {
  if (static_cast<std::size_t>(dv.size()) != state_->get_nv()) {
    throw_pretty(
        "Invalid argument: " << "dv has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + ")");
  }
  data->dv = dv;
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateForce(
    const std::shared_ptr<ContactDataMultiple>& data, const VectorXs& force) {
  if (static_cast<std::size_t>(force.size()) !=
      (compute_all_contacts_ ? nc_total_ : nc_)) {
    throw_pretty(
        "Invalid argument: "
        << "force has wrong dimension (it should be " +
               std::to_string((compute_all_contacts_ ? nc_total_ : nc_)) + ")");
  }
  if (static_cast<std::size_t>(data->contacts.size()) != contacts_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of contact datas and models");
  }

  for (ForceIterator it = data->fext.begin(); it != data->fext.end(); ++it) {
    *it = pinocchio::ForceTpl<Scalar>::Zero();
  }

  std::size_t nc = 0;
  typename ContactModelContainer::iterator it_m, end_m;
  typename ContactDataContainer::iterator it_d, end_d;
  if (compute_all_contacts_) {
    for (it_m = contacts_.begin(), end_m = contacts_.end(),
        it_d = data->contacts.begin(), end_d = data->contacts.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the contact name between data and model");
      const std::size_t nc_i = m_i->contact->get_nc();
      if (m_i->active) {
        const Eigen::VectorBlock<const VectorXs, Eigen::Dynamic> force_i =
            force.segment(nc, nc_i);
        m_i->contact->updateForce(d_i, force_i);
        const pinocchio::JointIndex joint =
            state_->get_pinocchio()->frames[d_i->frame].parentJoint;
        data->fext[joint] = d_i->fext;
      } else {
        m_i->contact->setZeroForce(d_i);
      }
      nc += nc_i;
    }
  } else {
    for (it_m = contacts_.begin(), end_m = contacts_.end(),
        it_d = data->contacts.begin(), end_d = data->contacts.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the contact name between data and model");
      if (m_i->active) {
        const std::size_t nc_i = m_i->contact->get_nc();
        const Eigen::VectorBlock<const VectorXs, Eigen::Dynamic> force_i =
            force.segment(nc, nc_i);
        m_i->contact->updateForce(d_i, force_i);
        const pinocchio::JointIndex joint =
            state_->get_pinocchio()->frames[d_i->frame].parentJoint;
        data->fext[joint] = d_i->fext;
        nc += nc_i;
      } else {
        m_i->contact->setZeroForce(d_i);
      }
    }
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateAccelerationDiff(
    const std::shared_ptr<ContactDataMultiple>& data,
    const MatrixXs& ddv_dx) const {
  if (static_cast<std::size_t>(ddv_dx.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(ddv_dx.cols()) != state_->get_ndx()) {
    throw_pretty(
        "Invalid argument: " << "ddv_dx has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + "," +
                                    std::to_string(state_->get_ndx()) + ")");
  }
  data->ddv_dx = ddv_dx;
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateForceDiff(
    const std::shared_ptr<ContactDataMultiple>& data, const MatrixXs& df_dx,
    const MatrixXs& df_du) const {
  const std::size_t ndx = state_->get_ndx();
  if (static_cast<std::size_t>(df_dx.rows()) !=
          (compute_all_contacts_ ? nc_total_ : nc_) ||
      static_cast<std::size_t>(df_dx.cols()) != ndx) {
    throw_pretty(
        "Invalid argument: "
        << "df_dx has wrong dimension (it should be " +
               std::to_string((compute_all_contacts_ ? nc_total_ : nc_)) + "," +
               std::to_string(ndx) + ")");
  }
  if (static_cast<std::size_t>(df_du.rows()) !=
          (compute_all_contacts_ ? nc_total_ : nc_) ||
      static_cast<std::size_t>(df_du.cols()) != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "df_du has wrong dimension (it should be " +
               std::to_string((compute_all_contacts_ ? nc_total_ : nc_)) + "," +
               std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(data->contacts.size()) != contacts_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of contact datas and models");
  }

  std::size_t nc = 0;
  typename ContactModelContainer::const_iterator it_m, end_m;
  typename ContactDataContainer::const_iterator it_d, end_d;
  if (compute_all_contacts_) {
    for (it_m = contacts_.begin(), end_m = contacts_.end(),
        it_d = data->contacts.begin(), end_d = data->contacts.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the contact name between data and model");
      const std::size_t nc_i = m_i->contact->get_nc();
      if (m_i->active) {
        const Eigen::Block<const MatrixXs> df_dx_i =
            df_dx.block(nc, 0, nc_i, ndx);
        const Eigen::Block<const MatrixXs> df_du_i =
            df_du.block(nc, 0, nc_i, nu_);
        m_i->contact->updateForceDiff(d_i, df_dx_i, df_du_i);
      } else {
        m_i->contact->setZeroForceDiff(d_i);
      }
      nc += nc_i;
    }
  } else {
    for (it_m = contacts_.begin(), end_m = contacts_.end(),
        it_d = data->contacts.begin(), end_d = data->contacts.end();
         it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the contact name between data and model");
      if (m_i->active) {
        const std::size_t nc_i = m_i->contact->get_nc();
        const Eigen::Block<const MatrixXs> df_dx_i =
            df_dx.block(nc, 0, nc_i, ndx);
        const Eigen::Block<const MatrixXs> df_du_i =
            df_du.block(nc, 0, nc_i, nu_);
        m_i->contact->updateForceDiff(d_i, df_dx_i, df_du_i);
        nc += nc_i;
      } else {
        m_i->contact->setZeroForceDiff(d_i);
      }
    }
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateRneaDiff(
    const std::shared_ptr<ContactDataMultiple>& data,
    pinocchio::DataTpl<Scalar>& pinocchio) const {
  if (static_cast<std::size_t>(data->contacts.size()) !=
      this->get_contacts().size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of contact datas and models");
  }
  typename ContactModelContainer::const_iterator it_m, end_m;
  typename ContactDataContainer::const_iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(),
      it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ContactItem>& m_i = it_m->second;
    const std::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first,
                  "it doesn't match the contact name between data and model");
    if (m_i->active) {
      switch (m_i->contact->get_type()) {
        case pinocchio::ReferenceFrame::LOCAL:
          break;
        case pinocchio::ReferenceFrame::WORLD:
        case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
          pinocchio.dtau_dq += d_i->dtau_dq;
          break;
      }
    }
  }
}

template <typename Scalar>
std::shared_ptr<ContactDataMultipleTpl<Scalar> >
ContactModelMultipleTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<ContactDataMultiple>(
      Eigen::aligned_allocator<ContactDataMultiple>(), this, data);
}

template <typename Scalar>
template <typename NewScalar>
ContactModelMultipleTpl<NewScalar> ContactModelMultipleTpl<Scalar>::cast()
    const {
  typedef ContactModelMultipleTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef ContactItemTpl<NewScalar> ContactType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), nu_);
  typename ContactModelContainer::const_iterator it_m, end_m;
  for (it_m = contacts_.begin(), end_m = contacts_.end(); it_m != end_m;
       ++it_m) {
    const std::string name = it_m->first;
    const ContactType& m_i = it_m->second->template cast<NewScalar>();
    ret.addContact(name, m_i.contact, m_i.active);
  }
  return ret;
}

template <typename Scalar>
const std::shared_ptr<StateMultibodyTpl<Scalar> >&
ContactModelMultipleTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename ContactModelMultipleTpl<Scalar>::ContactModelContainer&
ContactModelMultipleTpl<Scalar>::get_contacts() const {
  return contacts_;
}

template <typename Scalar>
std::size_t ContactModelMultipleTpl<Scalar>::get_nc() const {
  return nc_;
}

template <typename Scalar>
std::size_t ContactModelMultipleTpl<Scalar>::get_nc_total() const {
  return nc_total_;
}

template <typename Scalar>
std::size_t ContactModelMultipleTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
const std::set<std::string>& ContactModelMultipleTpl<Scalar>::get_active_set()
    const {
  return active_set_;
}

template <typename Scalar>
const std::set<std::string>& ContactModelMultipleTpl<Scalar>::get_inactive_set()
    const {
  return inactive_set_;
}

template <typename Scalar>
bool ContactModelMultipleTpl<Scalar>::getContactStatus(
    const std::string& name) const {
  typename ContactModelContainer::const_iterator it = contacts_.find(name);
  if (it != contacts_.end()) {
    return it->second->active;
  } else {
    std::cerr << "Warning: we couldn't get the status of the " << name
              << " contact item, it doesn't exist." << std::endl;
    return false;
  }
}

template <typename Scalar>
bool ContactModelMultipleTpl<Scalar>::getComputeAllContacts() const {
  return compute_all_contacts_;
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::setComputeAllContacts(const bool status) {
  compute_all_contacts_ = status;
}

template <class Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ContactModelMultipleTpl<Scalar>& model) {
  const auto& active = model.get_active_set();
  const auto& inactive = model.get_inactive_set();
  os << "ContactModelMultiple:" << std::endl;
  os << "  Active:" << std::endl;
  for (std::set<std::string>::const_iterator it = active.begin();
       it != active.end(); ++it) {
    const std::shared_ptr<
        typename ContactModelMultipleTpl<Scalar>::ContactItem>& contact_item =
        model.get_contacts().find(*it)->second;
    if (it != --active.end()) {
      os << "    " << *it << ": " << *contact_item << std::endl;
    } else {
      os << "    " << *it << ": " << *contact_item << std::endl;
    }
  }
  os << "  Inactive:" << std::endl;
  for (std::set<std::string>::const_iterator it = inactive.begin();
       it != inactive.end(); ++it) {
    const std::shared_ptr<
        typename ContactModelMultipleTpl<Scalar>::ContactItem>& contact_item =
        model.get_contacts().find(*it)->second;
    if (it != --inactive.end()) {
      os << "    " << *it << ": " << *contact_item << std::endl;
    } else {
      os << "    " << *it << ": " << *contact_item;
    }
  }
  return os;
}

}  // namespace crocoddyl
