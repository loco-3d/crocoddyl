///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_CONTACTS_HPP_
#define CROCODDYL_CORE_DATA_CONTACTS_HPP_

#include <boost/shared_ptr.hpp>
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {

struct DataCollectorContact : virtual DataCollectorAbstract {
  DataCollectorContact(boost::shared_ptr<ContactDataMultiple> contacts)
      : DataCollectorAbstract(), contacts(contacts) {}
  virtual ~DataCollectorContact() {}

  boost::shared_ptr<ContactDataMultiple> contacts;
};

struct DataCollectorMultibodyInContact : DataCollectorMultibody, DataCollectorContact {
  DataCollectorMultibodyInContact(pinocchio::Data* const pinocchio, boost::shared_ptr<ContactDataMultiple> contacts)
      : DataCollectorMultibody(pinocchio), DataCollectorContact(contacts) {}
  virtual ~DataCollectorMultibodyInContact() {}
};

struct DataCollectorActMultibodyInContact : DataCollectorMultibodyInContact, DataCollectorActuation {
  DataCollectorActMultibodyInContact(pinocchio::Data* const pinocchio,
                                     boost::shared_ptr<ActuationDataAbstract> actuation,
                                     boost::shared_ptr<ContactDataMultiple> contacts)
      : DataCollectorMultibodyInContact(pinocchio, contacts), DataCollectorActuation(actuation) {}
  virtual ~DataCollectorActMultibodyInContact() {}
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_IN_CONTACT_HPP_
