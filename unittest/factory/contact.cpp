///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "contact.hpp"

#include "crocoddyl/multibody/contacts/contact-1d.hpp"
#include "crocoddyl/multibody/contacts/contact-2d.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ContactModelTypes::Type> ContactModelTypes::all(
    ContactModelTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         const ContactModelTypes::Type& type) {
  switch (type) {
    case ContactModelTypes::ContactModel1D_LOCAL:
      os << "ContactModel1D_LOCAL";
      break;
    case ContactModelTypes::ContactModel1D_WORLD:
      os << "ContactModel1D_WORLD";
      break;
    case ContactModelTypes::ContactModel1D_LWA:
      os << "ContactModel1D_LWA";
      break;
    case ContactModelTypes::ContactModel2D:
      os << "ContactModel2D";
      break;
    case ContactModelTypes::ContactModel3D_LOCAL:
      os << "ContactModel3D_LOCAL";
      break;
    case ContactModelTypes::ContactModel3D_WORLD:
      os << "ContactModel3D_WORLD";
      break;
    case ContactModelTypes::ContactModel3D_LWA:
      os << "ContactModel3D_LWA";
      break;
    case ContactModelTypes::ContactModel6D_LOCAL:
      os << "ContactModel6D_LOCAL";
      break;
    case ContactModelTypes::ContactModel6D_WORLD:
      os << "ContactModel6D_WORLD";
      break;
    case ContactModelTypes::ContactModel6D_LWA:
      os << "ContactModel6D_LWA";
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

std::shared_ptr<crocoddyl::ContactModelAbstract> ContactModelFactory::create(
    ContactModelTypes::Type contact_type, PinocchioModelTypes::Type model_type,
    Eigen::Vector2d gains, const std::string frame_name, std::size_t nu) const {
  PinocchioModelFactory model_factory(model_type);
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::make_shared<crocoddyl::StateMultibody>(model_factory.create());
  std::shared_ptr<crocoddyl::ContactModelAbstract> contact;
  std::size_t frame_id = 0;
  if (frame_name == "") {
    frame_id = model_factory.get_frame_ids()[0];
  } else {
    frame_id = state->get_pinocchio()->getFrameId(frame_name);
  }
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (contact_type) {
    case ContactModelTypes::ContactModel1D_LOCAL: {
      pinocchio::SE3 M = pinocchio::SE3::Random();
      gains[0] =
          0;  // TODO(cmastalli): remove hard-coded zero when fixed the contact
      contact = std::make_shared<crocoddyl::ContactModel1D>(
          state, frame_id, 0., pinocchio::ReferenceFrame::LOCAL, M.rotation(),
          nu, gains);
      break;
    }
    case ContactModelTypes::ContactModel1D_WORLD: {
      pinocchio::SE3 M = pinocchio::SE3::Random();
      gains[0] =
          0;  // TODO(cmastalli): remove hard-coded zero when fixed the contact
      contact = std::make_shared<crocoddyl::ContactModel1D>(
          state, frame_id, 0., pinocchio::ReferenceFrame::WORLD, M.rotation(),
          nu, gains);
      break;
    }
    case ContactModelTypes::ContactModel1D_LWA: {
      pinocchio::SE3 M = pinocchio::SE3::Random();
      gains[0] =
          0;  // TODO(cmastalli): remove hard-coded zero when fixed the contact
      contact = std::make_shared<crocoddyl::ContactModel1D>(
          state, frame_id, 0., pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
          M.rotation(), nu, gains);
      break;
    }
    case ContactModelTypes::ContactModel2D:
      gains[0] =
          0;  // TODO(cmastalli): remove hard-coded zero when fixed the contact
      contact = std::make_shared<crocoddyl::ContactModel2D>(
          state, frame_id, Eigen::Vector2d::Zero(), nu, gains);
      break;
    case ContactModelTypes::ContactModel3D_LOCAL:
      contact = std::make_shared<crocoddyl::ContactModel3D>(
          state, frame_id, Eigen::Vector3d::Zero(),
          pinocchio::ReferenceFrame::LOCAL, nu, gains);
      break;
    case ContactModelTypes::ContactModel3D_WORLD:
      contact = std::make_shared<crocoddyl::ContactModel3D>(
          state, frame_id, Eigen::Vector3d::Zero(),
          pinocchio::ReferenceFrame::WORLD, nu, gains);
      break;
    case ContactModelTypes::ContactModel3D_LWA:
      contact = std::make_shared<crocoddyl::ContactModel3D>(
          state, frame_id, Eigen::Vector3d::Zero(),
          pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, nu, gains);
      break;
    case ContactModelTypes::ContactModel6D_LOCAL:
      contact = std::make_shared<crocoddyl::ContactModel6D>(
          state, frame_id, pinocchio::SE3::Identity(),
          pinocchio::ReferenceFrame::LOCAL, nu, gains);
      break;
    case ContactModelTypes::ContactModel6D_WORLD:
      contact = std::make_shared<crocoddyl::ContactModel6D>(
          state, frame_id, pinocchio::SE3::Identity(),
          pinocchio::ReferenceFrame::WORLD, nu, gains);
      break;
    case ContactModelTypes::ContactModel6D_LWA:
      contact = std::make_shared<crocoddyl::ContactModel6D>(
          state, frame_id, pinocchio::SE3::Identity(),
          pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, nu, gains);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ContactModelTypes::Type given");
      break;
  }
  return contact;
}

std::shared_ptr<crocoddyl::ContactModelAbstract> create_random_contact() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  std::shared_ptr<crocoddyl::ContactModelAbstract> contact;
  ContactModelFactory factory;
  if (rand() % 4 == 0) {
    contact = factory.create(ContactModelTypes::ContactModel1D_LOCAL,
                             PinocchioModelTypes::RandomHumanoid,
                             Eigen::Vector2d::Random());
  } else if (rand() % 4 == 1) {
    contact = factory.create(ContactModelTypes::ContactModel2D,
                             PinocchioModelTypes::RandomHumanoid,
                             Eigen::Vector2d::Random());
  } else if (rand() % 4 == 2) {
    contact = factory.create(ContactModelTypes::ContactModel3D_LOCAL,
                             PinocchioModelTypes::RandomHumanoid,
                             Eigen::Vector2d::Random());
  } else {
    contact = factory.create(ContactModelTypes::ContactModel6D_LOCAL,
                             PinocchioModelTypes::RandomHumanoid,
                             Eigen::Vector2d::Random());
  }
  return contact;
}

}  // namespace unittest
}  // namespace crocoddyl
