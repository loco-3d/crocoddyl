///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACTS_FACTORY_HPP_
#define CROCODDYL_CONTACTS_FACTORY_HPP_

#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/numdiff/contact.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct ContactModelTypes {
  enum Type {
    ContactModel1D_LOCAL,
    ContactModel1D_WORLD,
    ContactModel1D_LWA,
    ContactModel2D,
    ContactModel3D_LOCAL,
    ContactModel3D_WORLD,
    ContactModel3D_LWA,
    ContactModel6D_LOCAL,
    ContactModel6D_WORLD,
    ContactModel6D_LWA,
    NbContactModelTypes
  };
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

std::ostream& operator<<(std::ostream& os, const ContactModelTypes::Type& type);

class ContactModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ContactModelFactory();
  ~ContactModelFactory();

  std::shared_ptr<crocoddyl::ContactModelAbstract> create(
      ContactModelTypes::Type contact_type,
      PinocchioModelTypes::Type model_type, Eigen::Vector2d gains,
      const std::string frame_name = std::string(""),
      const std::size_t nu = std::numeric_limits<std::size_t>::max()) const;
};

std::shared_ptr<crocoddyl::ContactModelAbstract> create_random_contact();

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_CONTACTS_FACTORY_HPP_
