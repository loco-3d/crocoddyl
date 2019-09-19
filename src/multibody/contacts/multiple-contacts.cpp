///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {

ContactModelMultiple::ContactModelMultiple(StateMultibody& state, unsigned int const& nu)
    : state_(state), nc_(0), nu_(nu) {}

ContactModelMultiple::ContactModelMultiple(StateMultibody& state) : state_(state), nc_(0), nu_(state.get_nv()) {}

ContactModelMultiple::~ContactModelMultiple() {}

void ContactModelMultiple::addContact(const std::string& name, ContactModelAbstract* const contact) {
  assert(contact->get_nu() == nu_ && "Contact item doesn't have the same control dimension");
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

void ContactModelMultiple::calc(const boost::shared_ptr<ContactDataMultiple>& data,
                                const Eigen::Ref<const Eigen::VectorXd>& x) {
  assert(data->contacts.size() == contacts_.size() && "it doesn't match the number of contact datas and models");
  unsigned int nc = 0;

  unsigned int const& nv = state_.get_nv();
  ContactModelContainer::iterator it_m, end_m;
  ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the contact name between data and model");

    m_i.contact->calc(d_i, x);
    unsigned int const& nc_i = m_i.contact->get_nc();
    data->a0.segment(nc, nc_i) = d_i->a0;
    data->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
    nc += nc_i;
  }
}

void ContactModelMultiple::calcDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                    const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  assert(data->contacts.size() == contacts_.size() && "it doesn't match the number of contact datas and models");
  if (recalc) {
    calc(data, x);
  }
  unsigned int nc = 0;

  unsigned int const& ndx = state_.get_ndx();
  ContactModelContainer::iterator it_m, end_m;
  ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the contact name between data and model");

    m_i.contact->calcDiff(d_i, x, false);
    unsigned int const& nc_i = m_i.contact->get_nc();
    data->da_dx.block(nc, 0, nc_i, ndx) = d_i->da_dx;
    nc += nc_i;
  }
}

void ContactModelMultiple::updateAcceleration(const boost::shared_ptr<ContactDataMultiple>& data,
                                              const Eigen::VectorXd& dv) const {
  assert(dv.rows() == state_.get_nv() && "dv has wrong dimension");

  data->dv = dv;
}

void ContactModelMultiple::updateForce(const boost::shared_ptr<ContactDataMultiple>& data,
                                       const Eigen::VectorXd& force) {
  assert(force.size() == nc_ && "force has wrong dimension, it should be nc vector");
  assert(data->contacts.size() == contacts_.size() && "it doesn't match the number of contact datas and models");
  unsigned int nc = 0;

  for (ForceIterator it = data->fext.begin(); it != data->fext.end(); ++it) {
    *it = pinocchio::Force::Zero();
  }

  ContactModelContainer::iterator it_m, end_m;
  ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the contact name between data and model");

    unsigned int const& nc_i = m_i.contact->get_nc();
    const Eigen::VectorBlock<const Eigen::VectorXd, Eigen::Dynamic> force_i = force.segment(nc, nc_i);
    m_i.contact->updateForce(d_i, force_i);
    data->fext[d_i->joint] = d_i->f;
    nc += nc_i;
  }
}

void ContactModelMultiple::updateAccelerationDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                                  const Eigen::MatrixXd& ddv_dx) const {
  assert((ddv_dx.rows() == state_.get_nv() && ddv_dx.cols() == state_.get_ndx()) && "ddv_dx has wrong dimension");

  data->ddv_dx = ddv_dx;
}

void ContactModelMultiple::updateForceDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                           const Eigen::MatrixXd& df_dx, const Eigen::MatrixXd& df_du) const {
  unsigned int const& ndx = state_.get_ndx();
  assert((df_dx.rows() == nc_ || df_dx.cols() == ndx) && "df_dx has wrong dimension");
  assert((df_du.rows() == nc_ || df_du.cols() == nu_) && "df_du has wrong dimension");
  assert(data->contacts.size() == contacts_.size() && "it doesn't match the number of contact datas and models");
  unsigned int nc = 0;

  ContactModelContainer::const_iterator it_m, end_m;
  ContactDataContainer::const_iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    const boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the contact name between data and model");

    unsigned int const& nc_i = m_i.contact->get_nc();
    const Eigen::Block<const Eigen::MatrixXd> df_dx_i = df_dx.block(nc, 0, nc_i, ndx);
    const Eigen::Block<const Eigen::MatrixXd> df_du_i = df_du.block(nc, 0, nc_i, nu_);
    m_i.contact->updateForceDiff(d_i, df_dx_i, df_du_i);
    nc += nc_i;
  }
}

boost::shared_ptr<ContactDataMultiple> ContactModelMultiple::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactDataMultiple>(this, data);
}

StateMultibody& ContactModelMultiple::get_state() const { return state_; }

const ContactModelMultiple::ContactModelContainer& ContactModelMultiple::get_contacts() const { return contacts_; }

const unsigned int& ContactModelMultiple::get_nc() const { return nc_; }

const unsigned int& ContactModelMultiple::get_nu() const { return nu_; }

}  // namespace crocoddyl
