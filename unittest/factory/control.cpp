///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/make_shared.hpp>
#include "control.hpp"
#include "crocoddyl/core/controls/poly-zero.hpp"
#include "crocoddyl/core/controls/poly-one.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ControlTypes::Type> ControlTypes::all(ControlTypes::init_all());

std::ostream& operator<<(std::ostream& os, ControlTypes::Type type) {
  switch (type) {
    case ControlTypes::PolyZero:
      os << "PolyZero";
      break;
    case ControlTypes::PolyOne:
      os << "PolyOne";
      break;
    case ControlTypes::NbControlTypes:
      os << "NbControlTypes";
      break;
    default:
      break;
  }
  return os;
}

ControlFactory::ControlFactory() {}
ControlFactory::~ControlFactory() {}

boost::shared_ptr<crocoddyl::ControlAbstract> ControlFactory::create(ControlTypes::Type control_type,
                                                                     const std::size_t nu) const {
  boost::shared_ptr<crocoddyl::ControlAbstract> control;
  switch (control_type) {
    case ControlTypes::PolyZero:
      control = boost::make_shared<crocoddyl::ControlPolyZero>(nu);
      break;
    case ControlTypes::PolyOne:
      control = boost::make_shared<crocoddyl::ControlPolyOne>(nu);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ControlTypes::Type given");
      break;
  }
  return control;
}

}  // namespace unittest
}  // namespace crocoddyl
