///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {

ContactModelMultiple::ContactModelMultiple(boost::shared_ptr<StateMultibody> state, const std::size_t& nu)
    : state_(state), nc_(0), nu_(nu) {}

ContactModelMultiple::ContactModelMultiple(boost::shared_ptr<StateMultibody> state)
    : state_(state), nc_(0), nu_(state->get_nv()) {}

ContactModelMultiple::~ContactModelMultiple() {}

void ContactModelMultiple::addContact(const std::string& name, boost::shared_ptr<ContactModelAbstract> contact) {
  if (contact->get_nu() != nu_) {
    throw std::invalid_argument("contact item doesn't have the the same control dimension (" + to_string(nu_) + ")");
  }
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
  if (data->contacts.size() != contacts_.size()) {
    throw std::invalid_argument("it doesn't match the number of contact datas and models");
  }
  std::size_t nc = 0;

  const std::size_t& nv = state_->get_nv();
  ContactModelContainer::iterator it_m, end_m;
  ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the contact name between data and model");

    m_i.contact->calc(d_i, x);
    const std::size_t& nc_i = m_i.contact->get_nc();
    data->a0.segment(nc, nc_i) = d_i->a0;
    data->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
    nc += nc_i;
  }
}

void ContactModelMultiple::calcDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                    const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  if (data->contacts.size() != contacts_.size()) {
    throw std::invalid_argument("it doesn't match the number of contact datas and models");
  }
  if (recalc) {
    calc(data, x);
  }
  std::size_t nc = 0;

  const std::size_t& ndx = state_->get_ndx();
  ContactModelContainer::iterator it_m, end_m;
  ContactDataContainer::iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the contact name between data and model");

    m_i.contact->calcDiff(d_i, x, false);
    const std::size_t& nc_i = m_i.contact->get_nc();
    data->da0_dx.block(nc, 0, nc_i, ndx) = d_i->da0_dx;
    nc += nc_i;
  }
}

void ContactModelMultiple::updateAcceleration(const boost::shared_ptr<ContactDataMultiple>& data,
                                              const Eigen::VectorXd& dv) const {
  if (static_cast<std::size_t>(dv.size()) != state_->get_nv()) {
    throw std::invalid_argument("dv has wrong dimension (it should be " + to_string(state_->get_nv()) + ")");
  }
  data->dv = dv;
}

void ContactModelMultiple::updateForce(const boost::shared_ptr<ContactDataMultiple>& data,
                                       const Eigen::VectorXd& force) {
  if (static_cast<std::size_t>(force.size()) != nc_) {
    throw std::invalid_argument("force has wrong dimension (it should be " + to_string(nc_) + ")");
  }
  if (static_cast<std::size_t>(data->contacts.size()) != contacts_.size()) {
    throw std::invalid_argument("it doesn't match the number of contact datas and models");
  }
  std::size_t nc = 0;

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

    const std::size_t& nc_i = m_i.contact->get_nc();
    const Eigen::VectorBlock<const Eigen::VectorXd, Eigen::Dynamic> force_i = force.segment(nc, nc_i);
    m_i.contact->updateForce(d_i, force_i);
    data->fext[d_i->joint] = d_i->f;
    nc += nc_i;
  }
}

void ContactModelMultiple::updateAccelerationDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                                  const Eigen::MatrixXd& ddv_dx) const {
  if (static_cast<std::size_t>(ddv_dx.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(ddv_dx.cols()) != state_->get_ndx()) {
    throw std::invalid_argument("ddv_dx has wrong dimension (it should be " + to_string(state_->get_nv()) + "," +
                                to_string(state_->get_ndx()) + ")");
  }
  data->ddv_dx = ddv_dx;
}

void ContactModelMultiple::updateForceDiff(const boost::shared_ptr<ContactDataMultiple>& data,
                                           const Eigen::MatrixXd& df_dx, const Eigen::MatrixXd& df_du) const {
  const std::size_t& ndx = state_->get_ndx();
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ || static_cast<std::size_t>(df_dx.cols()) != ndx) {
    throw std::invalid_argument("df_dx has wrong dimension (it should be " + to_string(nc_) + "," + to_string(ndx) +
                                ")");
  }
  if (static_cast<std::size_t>(df_du.rows()) != nc_ || static_cast<std::size_t>(df_du.cols()) != nu_) {
    throw std::invalid_argument("df_du has wrong dimension (it should be " + to_string(nc_) + "," + to_string(nu_) +
                                ")");
  }
  if (static_cast<std::size_t>(data->contacts.size()) != contacts_.size()) {
    throw std::invalid_argument("it doesn't match the number of contact datas and models");
  }
  std::size_t nc = 0;

  ContactModelContainer::const_iterator it_m, end_m;
  ContactDataContainer::const_iterator it_d, end_d;
  for (it_m = contacts_.begin(), end_m = contacts_.end(), it_d = data->contacts.begin(), end_d = data->contacts.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ContactItem& m_i = it_m->second;
    const boost::shared_ptr<ContactDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the contact name between data and model");

    std::size_t const& nc_i = m_i.contact->get_nc();
    const Eigen::Block<const Eigen::MatrixXd> df_dx_i = df_dx.block(nc, 0, nc_i, ndx);
    const Eigen::Block<const Eigen::MatrixXd> df_du_i = df_du.block(nc, 0, nc_i, nu_);
    m_i.contact->updateForceDiff(d_i, df_dx_i, df_du_i);
    nc += nc_i;
  }
}

boost::shared_ptr<ContactDataMultiple> ContactModelMultiple::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactDataMultiple>(this, data);
}

const boost::shared_ptr<StateMultibody>& ContactModelMultiple::get_state() const { return state_; }

const ContactModelMultiple::ContactModelContainer& ContactModelMultiple::get_contacts() const { return contacts_; }

const std::size_t& ContactModelMultiple::get_nc() const { return nc_; }

const std::size_t& ContactModelMultiple::get_nu() const { return nu_; }

}  // namespace crocoddyl
