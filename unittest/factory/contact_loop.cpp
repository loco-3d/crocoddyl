///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "contact_loop.hpp"

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contacts/contact-6d-loop.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ContactLoopModelTypes::Type> ContactLoopModelTypes::all(
    ContactLoopModelTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         const ContactLoopModelTypes::Type& type) {
  switch (type) {
    case ContactLoopModelTypes::ContactModel6DLoop_LOCAL:
      os << "ContactLoopModel6D_LOCAL";
      break;
    case ContactLoopModelTypes::NbContactModelTypes:
      os << "NbContactModelTypes";
      break;
    default:
      os << "Unknown type";
      break;
  }
  return os;
}

ContactLoopModelFactory::ContactLoopModelFactory() {}
ContactLoopModelFactory::~ContactLoopModelFactory() {}

boost::shared_ptr<crocoddyl::ContactModelAbstract>
ContactLoopModelFactory::create(ContactLoopModelTypes::Type contact_type,
                                PinocchioModelTypes::Type model_type,
                                const int joint1_id,
                                const pinocchio::SE3& joint1_placement,
                                const int joint2_id,
                                const pinocchio::SE3& joint2_placement,
                                Eigen::Vector2d gains, std::size_t nu) const {
  PinocchioModelFactory model_factory(model_type);
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(model_factory.create());
  boost::shared_ptr<crocoddyl::ContactModelAbstract> contact;
  if (nu == std::numeric_limits<std::size_t>::max()) {
    nu = state->get_nv();
  }
  switch (contact_type) {
    case ContactLoopModelTypes::ContactModel6DLoop_LOCAL:
      contact = boost::make_shared<crocoddyl::ContactModel6DLoop>(
          state, joint1_id, joint1_placement, joint2_id, joint2_placement,
          pinocchio::ReferenceFrame::LOCAL, nu, gains);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ContactLoopModelTypes::Type given");
      break;
  }
  return contact;
}

boost::shared_ptr<crocoddyl::ContactModelAbstract>
create_random_loop_contact() {
  static bool once = true;
  if (once) {
    srand((unsigned)time(NULL));
    once = false;
  }
  boost::shared_ptr<crocoddyl::ContactModelAbstract> contact;
  ContactLoopModelFactory factory;
  ;
  if (rand() % 1 == 0) {
    contact = factory.create(
        ContactLoopModelTypes::ContactModel6DLoop_LOCAL,
        PinocchioModelTypes::RandomHumanoid, 0,
        pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Random()),
        1,
        pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Random()),
        Eigen::Vector2d::Random());
  }
  return contact;
}

}  // namespace unittest
}  // namespace crocoddyl
