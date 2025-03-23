///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_NAMESPACE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_NAMESPACE_HPP_

#include <boost/python.hpp>
#include <string>

namespace crocoddyl {
namespace python {

/**
 * @brief Get the current Python scope
 * @return the name of the current Python scope.
 */
inline std::string getCurrentScopeName() {
  namespace bp = boost::python;
  bp::scope current_scope;
  return std::string(bp::extract<const char *>(current_scope.attr("__name__")));
}

/**
 * @brief Get or create a Python scope
 *
 * @param submodule_name  name of the submodule
 * @return the submodule related to the namespace name.
 */
inline boost::python::object getOrCreatePythonNamespace(
    const std::string &submodule_name) {
  namespace bp = boost::python;
  const std::string complete_submodule_name =
      getCurrentScopeName() + "." + submodule_name;
  bp::object submodule(
      bp::borrowed(PyImport_AddModule(complete_submodule_name.c_str())));
  bp::scope current_scope;
  current_scope.attr(submodule_name.c_str()) = submodule;
  return submodule;
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_NAMESPACE_HPP_
