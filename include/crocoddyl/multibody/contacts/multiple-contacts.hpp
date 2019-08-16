///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_

#include <string>
#include <map>
#include <utility>
#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

struct ContactItem {
  ContactItem() {}
  ContactItem(const std::string& name, ContactModelAbstract* contact) : name(name), contact(contact) {}

  std::string name;
  ContactModelAbstract* contact;
};

class ContactModelMultiple : public ContactModelAbstract {
 public:
  typedef std::map<std::string, ContactItem> ContactModelContainer;
  typedef std::map<std::string, boost::shared_ptr<ContactDataAbstract> > ContactDataContainer;

  ContactModelMultiple(StateMultibody& state);
  ~ContactModelMultiple();

  void addContact(const std::string& name, ContactModelAbstract* const contact);
  void removeContact(const std::string& name);

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true);
  void updateLagrangian(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& lambda);
  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::Data* const data);

  const ContactModelContainer& get_contacts() const;
  const unsigned int& get_nc() const;

 private:
  ContactModelContainer contacts_;
  unsigned int nc_;
};

struct ContactDataMultiple : ContactDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ContactDataMultiple(Model* const model, pinocchio::Data* const data) : ContactDataAbstract(model, data) {
    for (ContactModelMultiple::ContactModelContainer::const_iterator it = model->get_contacts().begin();
         it != model->get_contacts().end(); ++it) {
      const ContactItem& item = it->second;
      contacts.insert(std::make_pair(item.name, item.contact->createData(data)));
    }
  }

  ContactModelMultiple::ContactDataContainer contacts;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
