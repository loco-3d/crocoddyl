///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "state.hpp"

#ifndef CROCODDYL_CONTACTS_FACTORY_HPP_
#define CROCODDYL_CONTACTS_FACTORY_HPP_

namespace crocoddyl_unit_test {

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
const std::vector<ContactModelTypes::Type> ContactModelTypes::all(ContactModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, const ContactModelTypes::Type& type) {
  switch (type) {
    case ContactModelTypes::ContactModel3D:
      os << "ContactModel3D";
      break;
    case ContactModelTypes::ContactModel6D:
      os << "ContactModel6D";
      break;
    case ContactModelTypes::NbContactModelTypes:
      os << "NbContactModelTypes";
      break;
    default:
      os << "Unknown type";
      break;
  }
  return os;
}

class ContactModelFactory {
 public:
  ContactModelFactory(ContactModelTypes::Type impulse_type, PinocchioModelTypes::Type model_type) {
    PinocchioModelFactory model_factory(model_type);
    StateFactory state_factory(StateTypes::StateMultibody, model_type);
    boost::shared_ptr<crocoddyl::StateMultibody> state =
        boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory.create());
    pinocchio_model_ = state->get_pinocchio();

    switch (impulse_type) {
      case ContactModelTypes::ContactModel3D:
        impulse_ = boost::make_shared<crocoddyl::ContactModel3D>(
            state, crocoddyl::FrameTranslation(model_factory.get_frame_id(), Eigen::Vector3d::Zero()));
        break;
      case ContactModelTypes::ContactModel6D:
        impulse_ = boost::make_shared<crocoddyl::ContactModel6D>(
            state, crocoddyl::FramePlacement(model_factory.get_frame_id(), pinocchio::SE3()));
        break;
      default:
        throw_pretty(__FILE__ ": Wrong ContactModelTypes::Type given");
        break;
    }
  }

  boost::shared_ptr<crocoddyl::ContactModelAbstract> create() const { return impulse_; }
  boost::shared_ptr<pinocchio::Model> get_pinocchio_model() const { return pinocchio_model_; }

 private:
  boost::shared_ptr<crocoddyl::ContactModelAbstract> impulse_;  //!< The pointer to the impulse model
  boost::shared_ptr<pinocchio::Model> pinocchio_model_;         //!< Pinocchio model
};

boost::shared_ptr<ContactModelFactory> create_random_factory() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  boost::shared_ptr<ContactModelFactory> ptr;
  if (rand() % 2 == 0) {
    ptr = boost::make_shared<ContactModelFactory>(ContactModelTypes::ContactModel3D,
                                                  PinocchioModelTypes::RandomHumanoid);
  } else {
    ptr = boost::make_shared<ContactModelFactory>(ContactModelTypes::ContactModel6D,
                                                  PinocchioModelTypes::RandomHumanoid);
  }
  return ptr;
}

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_CONTACTS_FACTORY_HPP_
