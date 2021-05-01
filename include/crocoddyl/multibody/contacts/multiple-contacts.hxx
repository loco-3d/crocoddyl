///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModelMultipleTpl<Scalar>::ContactModelMultipleTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nu)
    : state_(state), nc_(0), nc_total_(0), nu_(nu) {}

template <typename Scalar>
ContactModelMultipleTpl<Scalar>::ContactModelMultipleTpl(boost::shared_ptr<StateMultibody> state)
    : state_(state), nc_(0), nc_total_(0), nu_(state->get_nv()) {}

template <typename Scalar>
ContactModelMultipleTpl<Scalar>::~ContactModelMultipleTpl() {}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::addContact(const std::string& name,
                                                 boost::shared_ptr<ContactModelAbstract> contact, const bool active) {
  if (contact->get_nu() != nu_) {
    throw_pretty("Invalid argument: " << name
                                      << " contact item doesn't have the same control dimension (" +
                                             std::to_string(nu_) + ")");
  }
  std::pair<typename ContactModelContainer::iterator, bool> ret =
      contacts_.insert(std::make_pair(name, boost::make_shared<ContactItem>(name, contact, active)));
  if (ret.second == false) {
    std::cout << "Warning: we couldn't add the " << name << " contact item, it already existed." << std::endl;
  } else if (active) {
    nc_ += contact->get_nc();
    nc_total_ += contact->get_nc();
    std::vector<std::string>::iterator it =
        std::lower_bound(active_.begin(), active_.end(), name, std::less<std::string>());
    active_.insert(it, name);
  } else if (!active) {
    nc_total_ += contact->get_nc();
    std::vector<std::string>::iterator it =
        std::lower_bound(inactive_.begin(), inactive_.end(), name, std::less<std::string>());
    inactive_.insert(it, name);
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::removeContact(const std::string& name) {
  typename ContactModelContainer::iterator it = contacts_.find(name);
  if (it != contacts_.end()) {
    nc_ -= it->second->contact->get_nc();
    nc_total_ -= it->second->contact->get_nc();
    contacts_.erase(it);
    active_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
    inactive_.erase(std::remove(inactive_.begin(), inactive_.end(), name), inactive_.end());
  } else {
    std::cout << "Warning: we couldn't remove the " << name << " contact item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::changeContactStatus(const std::string& name, const bool active) {
  typename ContactModelContainer::iterator it = contacts_.find(name);
  if (it != contacts_.end()) {
    if (active && !it->second->active) {
      nc_ += it->second->contact->get_nc();
      std::vector<std::string>::iterator it =
          std::lower_bound(active_.begin(), active_.end(), name, std::less<std::string>());
      active_.insert(it, name);
      inactive_.erase(std::remove(inactive_.begin(), inactive_.end(), name), inactive_.end());
    } else if (!active && it->second->active) {
      nc_ -= it->second->contact->get_nc();
      active_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
      std::vector<std::string>::iterator it =
          std::lower_bound(inactive_.begin(), inactive_.end(), name, std::less<std::string>());
      inactive_.insert(it, name);
    }
    it->second->active = active;
  } else {
    std::cout << "Warning: we couldn't change the status of the " << name << " contact item, it doesn't exist."
              << std::endl;
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::calc(const boost::shared_ptr<ContactDataMultiple>& data,
                                           const Eigen::Ref<const VectorXs>& x) {
  if (data->contacts.size() != contacts_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of contact datas and models");
  }

  std::size_t nc = 0;
  const std::size_t nv = state_->get_nv();
  typename ContactModelContainer::iterator it_m, end_m;
  typename ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ContactItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the contact name between model and data ("
                                                    << it_m->first << " != " << it_d->first << ")");

      m_i->contact->calc(d_i, x);
      const std::size_t nc_i = m_i->contact->get_nc();
      data->a0.segment(nc, nc_i) = d_i->a0;
      data->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
      nc += nc_i;
    }
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                               const Eigen::Ref<const VectorXs>& x) {
  if (data->contacts.size() != contacts_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of contact datas and models");
  }

  std::size_t nc = 0;
  const std::size_t ndx = state_->get_ndx();
  typename ContactModelContainer::iterator it_m, end_m;
  typename ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ContactItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the contact name between model and data ("
                                                    << it_m->first << " != " << it_d->first << ")");

      m_i->contact->calcDiff(d_i, x);
      const std::size_t nc_i = m_i->contact->get_nc();
      data->da0_dx.block(nc, 0, nc_i, ndx) = d_i->da0_dx;
      nc += nc_i;
    }
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateAcceleration(const boost::shared_ptr<ContactDataMultiple>& data,
                                                         const VectorXs& dv) const {
  if (static_cast<std::size_t>(dv.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "dv has wrong dimension (it should be " + std::to_string(state_->get_nv()) + ")");
  }
  data->dv = dv;
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateForce(const boost::shared_ptr<ContactDataMultiple>& data,
                                                  const VectorXs& force) {
  if (static_cast<std::size_t>(force.size()) != nc_) {
    throw_pretty("Invalid argument: "
                 << "force has wrong dimension (it should be " + std::to_string(nc_) + ")");
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
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ContactItem>& m_i = it_m->second;
    const boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first, "it doesn't match the contact name between data and model");
    if (m_i->active) {
      const std::size_t nc_i = m_i->contact->get_nc();
      const Eigen::VectorBlock<const VectorXs, Eigen::Dynamic> force_i = force.segment(nc, nc_i);
      m_i->contact->updateForce(d_i, force_i);
      const pinocchio::JointIndex joint = state_->get_pinocchio()->frames[d_i->frame].parent;
      data->fext[joint] = d_i->f;
      nc += nc_i;
    } else {
      m_i->contact->setZeroForce(d_i);
    }
  }
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateAccelerationDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                                             const MatrixXs& ddv_dx) const {
  if (static_cast<std::size_t>(ddv_dx.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(ddv_dx.cols()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "ddv_dx has wrong dimension (it should be " + std::to_string(state_->get_nv()) + "," +
                        std::to_string(state_->get_ndx()) + ")");
  }
  data->ddv_dx = ddv_dx;
}

template <typename Scalar>
void ContactModelMultipleTpl<Scalar>::updateForceDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                                      const MatrixXs& df_dx, const MatrixXs& df_du) const {
  const std::size_t ndx = state_->get_ndx();
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ || static_cast<std::size_t>(df_dx.cols()) != ndx) {
    throw_pretty("Invalid argument: "
                 << "df_dx has wrong dimension (it should be " + std::to_string(nc_) + "," + std::to_string(ndx) +
                        ")");
  }
  if (static_cast<std::size_t>(df_du.rows()) != nc_ || static_cast<std::size_t>(df_du.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "df_du has wrong dimension (it should be " + std::to_string(nc_) + "," + std::to_string(nu_) +
                        ")");
  }
  if (static_cast<std::size_t>(data->contacts.size()) != contacts_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of contact datas and models");
  }

  std::size_t nc = 0;
  typename ContactModelContainer::const_iterator it_m, end_m;
  typename ContactDataContainer::const_iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ContactItem>& m_i = it_m->second;
    const boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first, "it doesn't match the contact name between data and model");
    if (m_i->active) {
      const std::size_t nc_i = m_i->contact->get_nc();
      const Eigen::Block<const MatrixXs> df_dx_i = df_dx.block(nc, 0, nc_i, ndx);
      const Eigen::Block<const MatrixXs> df_du_i = df_du.block(nc, 0, nc_i, nu_);
      m_i->contact->updateForceDiff(d_i, df_dx_i, df_du_i);
      nc += nc_i;
    } else {
      m_i->contact->setZeroForceDiff(d_i);
    }
  }
}

template <typename Scalar>
boost::shared_ptr<ContactDataMultipleTpl<Scalar> > ContactModelMultipleTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<ContactDataMultiple>(Eigen::aligned_allocator<ContactDataMultiple>(), this, data);
}

template <typename Scalar>
const boost::shared_ptr<StateMultibodyTpl<Scalar> >& ContactModelMultipleTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename ContactModelMultipleTpl<Scalar>::ContactModelContainer& ContactModelMultipleTpl<Scalar>::get_contacts()
    const {
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
const std::vector<std::string>& ContactModelMultipleTpl<Scalar>::get_active() const {
  return active_;
}

template <typename Scalar>
const std::vector<std::string>& ContactModelMultipleTpl<Scalar>::get_inactive() const {
  return inactive_;
}

template <typename Scalar>
bool ContactModelMultipleTpl<Scalar>::getContactStatus(const std::string& name) const {
  typename ContactModelContainer::const_iterator it = contacts_.find(name);
  if (it != contacts_.end()) {
    return it->second->active;
  } else {
    std::cout << "Warning: we couldn't get the status of the " << name << " contact item, it doesn't exist."
              << std::endl;
    return false;
  }
}

}  // namespace crocoddyl
