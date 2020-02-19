///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_CONTACTS_HPP_
#define CROCODDYL_CORE_DATA_CONTACTS_HPP_
#include "crocoddyl/multibody/fwd.hpp"
#include <boost/shared_ptr.hpp>
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {

template<typename Scalar>
struct DataCollectorContactTpl : virtual DataCollectorAbstractTpl<Scalar> {
  DataCollectorContactTpl<Scalar>(boost::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts)
      : DataCollectorAbstractTpl<Scalar>(), contacts(contacts) {}
  virtual ~DataCollectorContactTpl() {}

  boost::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts;
};

template<typename Scalar>
struct DataCollectorMultibodyInContactTpl : DataCollectorMultibodyTpl<Scalar>, DataCollectorContactTpl<Scalar> {
  DataCollectorMultibodyInContactTpl(pinocchio::DataTpl<Scalar>* const pinocchio, boost::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio), DataCollectorContactTpl<Scalar>(contacts) {}
  virtual ~DataCollectorMultibodyInContactTpl() {}
};

template<typename Scalar>
struct DataCollectorActMultibodyInContactTpl : DataCollectorMultibodyInContactTpl<Scalar>, DataCollectorActuationTpl<Scalar> {
  DataCollectorActMultibodyInContactTpl(pinocchio::DataTpl<Scalar>* const pinocchio,
                                     boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
                                     boost::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts)
      : DataCollectorMultibodyInContactTpl<Scalar>(pinocchio, contacts), DataCollectorActuationTpl<Scalar>(actuation) {}
  virtual ~DataCollectorActMultibodyInContactTpl() {}
};

  
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_IN_CONTACT_HPP_
