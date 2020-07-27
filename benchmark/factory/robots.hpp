///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_FACTORY_ROBOTS_HPP_
#define CROCODDYL_FACTORY_ROBOTS_HPP_

struct RobotBaseNames
{
  RobotBaseNames(contact_names, urdf_path,
                 srdf_path, ee_name, reference_conf):
    contact_names(contact_names),
    urdf_path(urdf_path),
    srdf_path(srdf_path),
    ee_name(ee_name),
    reference_conf(reference_conf){}

  std::vector<std::string> contact_names;
  std::string urdf_path;
  std::string srdf_path;
  std::string ee_name;
  std::string reference_conf;
};
