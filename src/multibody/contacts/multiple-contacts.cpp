///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {

ContactModelMultiple::ContactModelMultiple(StateMultibody& state)
    : ContactModelAbstract(state, (unsigned int)0), nc_(0) {}

ContactModelMultiple::~ContactModelMultiple() {}

void ContactModelMultiple::addContact(const std::string& name, ContactModelAbstract* const contact) {
  std::pair<ContactModelContainer::iterator, bool> ret =
      contacts_.insert(std::make_pair(name, ContactItem(name, contact)));
  if (ret.second == false) {
    std::cout << "Warning: this contact item already existed, we cannot add it" << std::endl;
  } else {
    nc_ += contact->get_nc();
  }
}

void ContactModelMultiple::removeContact(const std::string& name) {
  ContactModelContainer::iterator it = contacts_.find(name);
  if (it != contacts_.end()) {
    nc_ -= it->second.contact->get_nc();
    contacts_.erase(it);
  } else {
    std::cout << "Warning: this contact item doesn't exist, we cannot remove it" << std::endl;
  }
}

void ContactModelMultiple::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                const Eigen::Ref<const Eigen::VectorXd>& x) {
  ContactDataMultiple* d = static_cast<ContactDataMultiple*>(data.get());
  unsigned int nc = 0;

  unsigned int const& nv = state_.get_nv();
  ContactModelContainer::iterator it_m, end_m;
  ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = d->contacts.begin(), end_d = d->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;

    m_i.contact->calc(d_i, x);
    unsigned int const& nc_i = m_i.contact->get_nc();
    d->a0.segment(nc, nc_i) = d_i->a0;
    d->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
    nc += nc_i;
  }
}

void ContactModelMultiple::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                    const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  if (recalc) {
    calc(data, x);
  }
  ContactDataMultiple* d = static_cast<ContactDataMultiple*>(data.get());
  unsigned int nc = 0;

  unsigned int const& ndx = state_.get_ndx();
  ContactModelContainer::iterator it_m, end_m;
  ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = d->contacts.begin(), end_d = d->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;

    m_i.contact->calcDiff(d_i, x, false);
    unsigned int const& nc_i = m_i.contact->get_nc();
    d->Ax.block(nc, 0, nc_i, ndx) = d_i->Ax;
    nc += nc_i;
  }
}

void ContactModelMultiple::updateLagrangian(const boost::shared_ptr<ContactDataAbstract>& data,
                                            const Eigen::VectorXd& lambda) {
  assert(lambda.size() == nc_ &&
         "ContactModelMultiple::updateLagrangian: lambda has wrong dimension, it should be nc vector");
  ContactDataMultiple* d = static_cast<ContactDataMultiple*>(data.get());
  unsigned int nc = 0;

  for (ForceIterator it = d->fext.begin(); it != d->fext.end(); ++it) {
    *it = pinocchio::Force::Zero();
  }

  ContactModelContainer::iterator it_m, end_m;
  ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = d->contacts.begin(), end_d = d->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;

    unsigned int const& nc_i = m_i.contact->get_nc();
    m_i.contact->updateLagrangian(d_i, lambda.segment(nc, nc_i));
    d->fext[d_i->joint] = d_i->f;
    nc += nc_i;
  }
}

boost::shared_ptr<ContactDataAbstract> ContactModelMultiple::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactDataMultiple>(this, data);
}

const ContactModelMultiple::ContactModelContainer& ContactModelMultiple::get_contacts() const { return contacts_; }

const unsigned int& ContactModelMultiple::get_nc() const { return nc_; }

}  // namespace crocoddyl
