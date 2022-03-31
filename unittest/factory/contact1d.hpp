///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACT_1D_FACTORY_HPP_
#define CROCODDYL_CONTACT_1D_FACTORY_HPP_

#include <iostream>
#include <limits>

#include "state.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/numdiff/contact.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {
namespace unittest {

struct ContactModelMaskTypes {
  enum Type { X, Y, Z, NbMaskTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbMaskTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

struct PinocchioReferenceTypes {
  enum Type { LOCAL, WORLD, LOCAL_WORLD_ALIGNED, NbPinRefTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbPinRefTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, const ContactModelMaskTypes::Type& type);

std::ostream& operator<<(std::ostream& os, const PinocchioReferenceTypes::Type& type);

class ContactModel1DFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ContactModel1DFactory();
  ~ContactModel1DFactory();

  boost::shared_ptr<crocoddyl::ContactModelAbstract> create(
      ContactModelMaskTypes::Type mask_type, PinocchioModelTypes::Type model_type,
      PinocchioReferenceTypes::Type reference_type, const std::string frame_name = std::string(""),
      const std::size_t nu = std::numeric_limits<std::size_t>::max()) const;
};

boost::shared_ptr<crocoddyl::ContactModelAbstract> create_random_contact1d();

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_CONTACT_1D_FACTORY_HPP_
