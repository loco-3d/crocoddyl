///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, University of Edinburgh, Heriot-Watt University,
//                          LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACTS_LOOP_FACTORY_HPP_
#define CROCODDYL_CONTACTS_LOOP_FACTORY_HPP_

#include <iostream>
#include <limits>

#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/numdiff/contact.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct ContactLoopModelTypes {
  enum Type { ContactModel6DLoop_LOCAL, NbContactModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbContactModelTypes);
    for (int i = 0; i < NbContactModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os,
                         const ContactLoopModelTypes::Type& type);

class ContactLoopModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ContactLoopModelFactory();
  ~ContactLoopModelFactory();

  boost::shared_ptr<crocoddyl::ContactModelAbstract> create(
      ContactLoopModelTypes::Type contact_type,
      PinocchioModelTypes::Type model_type, const int joint1_id,
      const pinocchio::SE3& joint1_placement, const int joint2_id,
      const pinocchio::SE3& joint2_placement, Eigen::Vector2d gains,
      std::size_t nu = std::numeric_limits<std::size_t>::max()) const;
};

boost::shared_ptr<crocoddyl::ContactModelAbstract> create_random_loop_contact();

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_CONTACTS_LOOP_FACTORY_HPP_
