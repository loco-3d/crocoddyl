///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_FACTORY_ROBOT_EE_NAMES_HPP_
#define CROCODDYL_FACTORY_ROBOT_EE_NAMES_HPP_

#include "crocoddyl/multibody/fwd.hpp"

struct RobotEENames {
  RobotEENames(const std::string& robot_name,
               const std::vector<std::string>& contact_names,
               const std::vector<crocoddyl::ContactType>& contact_types,
               const std::string& urdf_path, const std::string& srdf_path,
               const std::string& ee_name, const std::string& reference_conf)
      : robot_name(robot_name),
        contact_names(contact_names),
        contact_types(contact_types),
        urdf_path(urdf_path),
        srdf_path(srdf_path),
        ee_name(ee_name),
        reference_conf(reference_conf) {}
  std::string robot_name;
  std::vector<std::string> contact_names;
  std::vector<crocoddyl::ContactType> contact_types;

  std::string urdf_path;
  std::string srdf_path;
  std::string ee_name;
  std::string reference_conf;
};

#endif  // CROCODDYL_FACTORY_ROBOT_EE_NAMES_HPP_
