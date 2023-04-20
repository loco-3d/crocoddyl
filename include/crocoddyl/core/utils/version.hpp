///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_VERSION_HPP_
#define CROCODDYL_CORE_UTILS_VERSION_HPP_

#include <sstream>
#include <string>

#include "crocoddyl/config.hh"

namespace crocoddyl {

///
/// \brief Returns the current version of Crocoddyl as a string using
///        the following standard:
///        CROCODDYL_MINOR_VERSION.CROCODDYL_MINOR_VERSION.CROCODDYL_PATCH_VERSION
///
inline std::string printVersion(const std::string& delimiter = ".") {
  std::ostringstream oss;
  oss << CROCODDYL_MAJOR_VERSION << delimiter << CROCODDYL_MINOR_VERSION
      << delimiter << CROCODDYL_PATCH_VERSION;
  return oss.str();
}

///
/// \brief Checks if the current version of Crocoddyl is at least the version
/// provided
///        by the input arguments.
///
/// \param[in] major_version Major version to check.
/// \param[in] minor_version Minor version to check.
/// \param[in] patch_version Patch version to check.
///
/// \returns true if the current version of Pinocchio is greater than the
/// version provided
///        by the input arguments.
///
inline bool checkVersionAtLeast(int major_version, int minor_version,
                                int patch_version) {
  return CROCODDYL_MAJOR_VERSION > major_version ||
         (CROCODDYL_MAJOR_VERSION >= major_version &&
          (CROCODDYL_MINOR_VERSION > minor_version ||
           (CROCODDYL_MINOR_VERSION >= minor_version &&
            CROCODDYL_PATCH_VERSION >= patch_version)));
}
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_VERSION_HPP_
