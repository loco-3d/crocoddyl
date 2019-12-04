///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_MULTIBODY_IN_CONTACT_HPP_
#define CROCODDYL_CORE_DATA_MULTIBODY_IN_CONTACT_HPP_

#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {

struct DataCollectorMultibodyInContact : DataCollectorMultibody {
  DataCollectorMultibodyInContact(pinocchio::Data* const data, ContactDataMultiple* const contacts)
      : DataCollectorMultibody(data), contacts(contacts) {}
  virtual ~DataCollectorMultibodyInContact() {}

  ContactDataMultiple* contacts;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_IN_CONTACT_HPP_