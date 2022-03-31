///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "contact1d.hpp"
#include "crocoddyl/multibody/contacts/contact-1d.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ContactModelMaskTypes::Type> ContactModelMaskTypes::all(ContactModelMaskTypes::init_all());

const std::vector<PinocchioReferenceTypes::Type> PinocchioReferenceTypes::all(PinocchioReferenceTypes::init_all());

std::ostream& operator<<(std::ostream& os, const ContactModelMaskTypes::Type& type) {
  switch (type) {
    case ContactModelMaskTypes::X:
      os << "X";
      break;
    case ContactModelMaskTypes::Y:
      os << "Y";
      break;
    case ContactModelMaskTypes::Z:
      os << "Z";
      break;
    case ContactModelMaskTypes::NbMaskTypes:
      os << "NbMaskTypes";
      break;
    default:
      os << "Unknown type";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const PinocchioReferenceTypes::Type& type) {
  switch (type) {
    case PinocchioReferenceTypes::LOCAL:
      os << "LOCAL";
      break;
    case PinocchioReferenceTypes::WORLD:
      os << "WORLD";
      break;
    case PinocchioReferenceTypes::LOCAL_WORLD_ALIGNED:
      os << "LOCAL_WORLD_ALIGNED";
      break;
    case PinocchioReferenceTypes::NbPinRefTypes:
      os << "NbPinRefTypes";
      break;
    default:
      os << "Unknown type";
      break;
  }
  return os;
}

ContactModel1DFactory::ContactModel1DFactory() {}
ContactModel1DFactory::~ContactModel1DFactory() {}

boost::shared_ptr<crocoddyl::ContactModelAbstract> ContactModel1DFactory::create(
    ContactModelMaskTypes::Type mask_type, PinocchioModelTypes::Type model_type,
    PinocchioReferenceTypes::Type reference_type, const std::string frame_name, std::size_t nu) const {
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

  Eigen::Vector2d gains = Eigen::Vector2d::Zero();
  switch (mask_type) {
    case ContactModelMaskTypes::X: {
      if (reference_type == PinocchioReferenceTypes::LOCAL) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::x,
                                                                pinocchio::LOCAL);
      } else if (reference_type == PinocchioReferenceTypes::WORLD) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::x,
                                                                pinocchio::WORLD);
      } else if (reference_type == PinocchioReferenceTypes::LOCAL_WORLD_ALIGNED) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::x,
                                                                pinocchio::LOCAL_WORLD_ALIGNED);
      }
      break;
    }
    case ContactModelMaskTypes::Y: {
      if (reference_type == PinocchioReferenceTypes::LOCAL) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::y,
                                                                pinocchio::LOCAL);
      } else if (reference_type == PinocchioReferenceTypes::WORLD) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::y,
                                                                pinocchio::WORLD);
      } else if (reference_type == PinocchioReferenceTypes::LOCAL_WORLD_ALIGNED) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::y,
                                                                pinocchio::LOCAL_WORLD_ALIGNED);
      }
      break;
    }
    case ContactModelMaskTypes::Z:
      if (reference_type == PinocchioReferenceTypes::LOCAL) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::z,
                                                                pinocchio::LOCAL);
      } else if (reference_type == PinocchioReferenceTypes::WORLD) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::z,
                                                                pinocchio::WORLD);
      } else if (reference_type == PinocchioReferenceTypes::LOCAL_WORLD_ALIGNED) {
        contact = boost::make_shared<crocoddyl::ContactModel1D>(state, frame_id, 0., nu, gains, Vector3MaskType::z,
                                                                pinocchio::LOCAL_WORLD_ALIGNED);
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ContactModelMaskTypes::Type given");
      break;
  }
  return contact;
}

boost::shared_ptr<crocoddyl::ContactModelAbstract> create_random_contact1d() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  boost::shared_ptr<crocoddyl::ContactModelAbstract> contact;
  ContactModel1DFactory factory;
  if (rand() % 3 == 0) {
    contact =
        factory.create(ContactModelMaskTypes::X, PinocchioModelTypes::RandomHumanoid, PinocchioReferenceTypes::LOCAL);
  } else if (rand() % 3 == 1) {
    contact =
        factory.create(ContactModelMaskTypes::Y, PinocchioModelTypes::RandomHumanoid, PinocchioReferenceTypes::LOCAL);
  } else {
    contact =
        factory.create(ContactModelMaskTypes::Z, PinocchioModelTypes::RandomHumanoid, PinocchioReferenceTypes::LOCAL);
  }
  return contact;
}

}  // namespace unittest
}  // namespace crocoddyl
