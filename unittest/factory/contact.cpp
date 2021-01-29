///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "contact.hpp"
#include "crocoddyl/multibody/contacts/contact-2d.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ContactModelTypes::Type> ContactModelTypes::all(ContactModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, const ContactModelTypes::Type& type) {
  switch (type) {
    case ContactModelTypes::ContactModel2D:
      os << "ContactModel2D";
      break;
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

ContactModelFactory::ContactModelFactory() {}
ContactModelFactory::~ContactModelFactory() {}

boost::shared_ptr<crocoddyl::ContactModelAbstract> ContactModelFactory::create(ContactModelTypes::Type contact_type,
                                                                               PinocchioModelTypes::Type model_type,
                                                                               const std::string frame_name,
                                                                               std::size_t nu) const {
  PinocchioModelFactory model_factory(model_type);
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(model_factory.create());
  boost::shared_ptr<crocoddyl::ContactModelAbstract> contact;
  std::size_t frame_id = 0;
  if (frame_name == "") {
    frame_id = model_factory.get_frame_id();
  } else {
    frame_id = state->get_pinocchio()->getFrameId(frame_name);
  }
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (contact_type) {
    case ContactModelTypes::ContactModel2D:
      contact = boost::make_shared<crocoddyl::ContactModel2D>(
          state, crocoddyl::FrameTranslation(frame_id, Eigen::Vector3d::Zero()), nu);
      break;
    case ContactModelTypes::ContactModel3D:
      contact = boost::make_shared<crocoddyl::ContactModel3D>(
          state, crocoddyl::FrameTranslation(frame_id, Eigen::Vector3d::Zero()), nu);
      break;
    case ContactModelTypes::ContactModel6D:
      contact = boost::make_shared<crocoddyl::ContactModel6D>(
          state, crocoddyl::FramePlacement(frame_id, pinocchio::SE3()), nu);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ContactModelTypes::Type given");
      break;
  }
  return contact;
}

boost::shared_ptr<crocoddyl::ContactModelAbstract> create_random_contact() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  boost::shared_ptr<crocoddyl::ContactModelAbstract> contact;
  ContactModelFactory factory;
  if (rand() % 3 == 0) {
    contact = factory.create(ContactModelTypes::ContactModel2D, PinocchioModelTypes::RandomHumanoid);
  } else if (rand() % 3 == 1) {
    contact = factory.create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::RandomHumanoid);
  } else {
    contact = factory.create(ContactModelTypes::ContactModel6D, PinocchioModelTypes::RandomHumanoid);
  }
  return contact;
}

}  // namespace unittest
}  // namespace crocoddyl
