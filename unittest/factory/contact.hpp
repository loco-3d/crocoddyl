///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACTS_FACTORY_HPP_
#define CROCODDYL_CONTACTS_FACTORY_HPP_

#include <iostream>

#include "state.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/numdiff/contact.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {
namespace unittest {

struct ContactModelTypes {
  enum Type { ContactModel3D, ContactModel6D, NbContactModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbContactModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, const ContactModelTypes::Type& type);

class ContactModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ContactModelFactory();
  ~ContactModelFactory();

  boost::shared_ptr<crocoddyl::ContactModelAbstract> create(ContactModelTypes::Type contact_type,
                                                            PinocchioModelTypes::Type model_type) const;
};

boost::shared_ptr<crocoddyl::ContactModelAbstract> create_random_contact();

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_CONTACTS_FACTORY_HPP_
