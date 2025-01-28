///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2024, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/constraints/constraint-manager.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/map-converter.hpp"
#include "python/crocoddyl/utils/printable.hpp"
#include "python/crocoddyl/utils/set-converter.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(
    ConstraintModelManager_addConstraint_wrap,
    ConstraintModelManager::addConstraint, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ConstraintDataManager_resizeConst_wrap,
                                       ConstraintDataManager::resize, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ConstraintDataManager_resize_wrap,
                                       ConstraintDataManager::resize, 2, 3)

void exposeConstraintManager() {
  // Register custom converters between std::map and Python dict
  typedef std::shared_ptr<ConstraintItem> ConstraintItemPtr;
  typedef std::shared_ptr<ConstraintDataAbstract> ConstraintDataPtr;
  StdMapPythonVisitor<
      std::string, ConstraintItemPtr, std::less<std::string>,
      std::allocator<std::pair<const std::string, ConstraintItemPtr> >,
      true>::expose("StdMap_ConstraintItem");
  StdMapPythonVisitor<
      std::string, ConstraintDataPtr, std::less<std::string>,
      std::allocator<std::pair<const std::string, ConstraintDataPtr> >,
      true>::expose("StdMap_ConstraintData");

  bp::register_ptr_to_python<std::shared_ptr<ConstraintItem> >();

  bp::class_<ConstraintItem>(
      "ConstraintItem", "Describe a constraint item.\n\n",
      bp::init<std::string, std::shared_ptr<ConstraintModelAbstract>,
               bp::optional<bool> >(
          bp::args("self", "name", "constraint", "active"),
          "Initialize the constraint item.\n\n"
          ":param name: constraint name\n"
          ":param constraint: constraint model\n"
          ":param active: True if the constraint is activated (default true)"))
      .def_readwrite("name", &ConstraintItem::name, "constraint name")
      .add_property(
          "constraint",
          bp::make_getter(&ConstraintItem::constraint,
                          bp::return_value_policy<bp::return_by_value>()),
          "constraint model")
      .def_readwrite("active", &ConstraintItem::active, "constraint status")
      .def(CopyableVisitor<ConstraintItem>())
      .def(PrintableVisitor<ConstraintItem>());

  bp::register_ptr_to_python<std::shared_ptr<ConstraintModelManager> >();

  bp::class_<ConstraintModelManager>(
      "ConstraintModelManager",
      bp::init<std::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the total constraint model.\n\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the total constraint model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<std::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the total constraint model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state description"))
      .def("addConstraint", &ConstraintModelManager::addConstraint,
           ConstraintModelManager_addConstraint_wrap(
               bp::args("self", "name", "constraint", "active"),
               "Add a constraint item.\n\n"
               ":param name: constraint name\n"
               ":param constraint: constraint model\n"
               ":param active: True if the constraint is activated (default "
               "true)"))
      .def("removeConstraint", &ConstraintModelManager::removeConstraint,
           bp::args("self", "name"),
           "Remove a constraint item.\n\n"
           ":param name: constraint name")
      .def("changeConstraintStatus",
           &ConstraintModelManager::changeConstraintStatus,
           bp::args("self", "name", "active"),
           "Change the constraint status.\n\n"
           ":param name: constraint name\n"
           ":param active: constraint status (true for active and false for "
           "inactive)")
      .def<void (ConstraintModelManager::*)(
          const std::shared_ptr<ConstraintDataManager>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelManager::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the total constraint.\n\n"
          ":param data: constraint-manager data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ConstraintModelManager::*)(
          const std::shared_ptr<ConstraintDataManager>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelManager::calc, bp::args("self", "data", "x"),
          "Compute the total constraint value for nodes that depends only on "
          "the state.\n\n"
          "It updates the total constraint based on the state only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: constraint-manager data\n"
          ":param x: state point (dim. state.nx)")
      .def<void (ConstraintModelManager::*)(
          const std::shared_ptr<ConstraintDataManager>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelManager::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the total constraint.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: constraint-manager data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)\n")
      .def<void (ConstraintModelManager::*)(
          const std::shared_ptr<ConstraintDataManager>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelManager::calcDiff,
          bp::args("self", "data", "x"),
          "Compute the Jacobian of the total constraint for nodes that depends "
          "on the state only.\n\n"
          "It updates the Jacobian of the total constraint based on the state "
          "only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: constraint-manager data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &ConstraintModelManager::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the total constraint data.\n\n"
           ":param data: shared data\n"
           ":return total constraint data.")
      .add_property(
          "state",
          bp::make_function(&ConstraintModelManager::get_state,
                            bp::return_value_policy<bp::return_by_value>()),
          "state description")
      .add_property(
          "constraints",
          bp::make_function(&ConstraintModelManager::get_constraints,
                            bp::return_value_policy<bp::return_by_value>()),
          "stack of constraints")
      .add_property("nu", bp::make_function(&ConstraintModelManager::get_nu),
                    "dimension of control vector")
      .add_property("ng", bp::make_function(&ConstraintModelManager::get_ng),
                    "number of active inequality constraints")
      .add_property("nh", bp::make_function(&ConstraintModelManager::get_nh),
                    "number of active equality constraints")
      .add_property("ng_T",
                    bp::make_function(&ConstraintModelManager::get_ng_T),
                    "number of active inequality terminal constraints")
      .add_property("nh_T",
                    bp::make_function(&ConstraintModelManager::get_nh_T),
                    "number of active equality terminal constraints")
      .add_property(
          "active_set",
          bp::make_function(&ConstraintModelManager::get_active_set,
                            bp::return_value_policy<bp::return_by_value>()),
          "name of the active set of constraint items")
      .add_property(
          "inactive_set",
          bp::make_function(&ConstraintModelManager::get_inactive_set,
                            bp::return_value_policy<bp::return_by_value>()),
          "name of the inactive set of constraint items")
      .add_property("g_lb",
                    bp::make_function(&ConstraintModelManager::get_lb,
                                      bp::return_internal_reference<>()),
                    "lower bound of the inequality constraints")
      .add_property("g_ub",
                    bp::make_function(&ConstraintModelManager::get_ub,
                                      bp::return_internal_reference<>()),
                    "upper bound of the inequality constraints")
      .def("getConstraintStatus", &ConstraintModelManager::getConstraintStatus,
           bp::args("self", "name"),
           "Return the constraint status of a given constraint name.\n\n"
           ":param name: constraint name")
      .def(CopyableVisitor<ConstraintModelManager>())
      .def(PrintableVisitor<ConstraintModelManager>());

  bp::register_ptr_to_python<std::shared_ptr<ConstraintDataManager> >();

  bp::class_<ConstraintDataManager>(
      "ConstraintDataManager", "Class for total constraint data.\n\n",
      bp::init<ConstraintModelManager*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create total constraint data.\n\n"
          ":param model: total constraint model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 3>()])
      .def(
          "shareMemory",
          &ConstraintDataManager::shareMemory<DifferentialActionDataAbstract>,
          bp::args("self", "data"),
          "Share memory with a given differential action data\n\n"
          ":param model: differential action data that we want to share memory")
      .def("shareMemory",
           &ConstraintDataManager::shareMemory<ActionDataAbstract>,
           bp::args("self", "data"),
           "Share memory with a given action data\n\n"
           ":param model: action data that we want to share memory")
      .def("resize", &ConstraintDataManager::resize<ConstraintModelManager>,
           bp::with_custodian_and_ward_postcall<0, 1>(),
           ConstraintDataManager_resizeConst_wrap(
               bp::args("self", "model", "running_node"),
               "Resize the data given differential action data\n\n"
               ":param model: constraint manager model that defines how to "
               "resize "
               "the data\n"
               ":param running_node: true if we are resizing for a running "
               "node, false for terminal ones (default true)"))
      .def("resize",
           &ConstraintDataManager::resize<DifferentialActionModelAbstract,
                                          DifferentialActionDataAbstract>,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           ConstraintDataManager_resize_wrap(
               bp::args("self", "model", "data", "running_node"),
               "Resize the data given differential action data\n\n"
               ":param model: differential action model that defines how to "
               "resize "
               "the data\n"
               ":param data: differential action data that we want to resize\n"
               ":param running_node: true if we are resizing for a running "
               "node, false for terminal ones (default true)"))
      .def(
          "resize",
          &ConstraintDataManager::resize<ActionModelAbstract,
                                         ActionDataAbstract>,
          bp::with_custodian_and_ward_postcall<0, 2>(),
          ConstraintDataManager_resize_wrap(
              bp::args("self", "model", "data", "running_node"),
              "Resize the data given action data\n\n"
              ":param model: action model that defines how to resize the data\n"
              ":param data: action data that we want to resize\n"
              ":param running_node: true if we are resizing for a running "
              "node, false for terminal ones (default true)"))
      .add_property(
          "constraints",
          bp::make_getter(&ConstraintDataManager::constraints,
                          bp::return_value_policy<bp::return_by_value>()),
          "stack of constraints data")
      .add_property("shared",
                    bp::make_getter(&ConstraintDataManager::shared,
                                    bp::return_internal_reference<>()),
                    "shared data")
      .add_property(
          "g",
          bp::make_function(&ConstraintDataManager::get_g,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ConstraintDataManager::set_g),
          "Inequality constraint residual")
      .add_property(
          "Gx",
          bp::make_function(&ConstraintDataManager::get_Gx,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ConstraintDataManager::set_Gx),
          "Jacobian of the inequality constraint")
      .add_property(
          "Gu",
          bp::make_function(&ConstraintDataManager::get_Gu,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ConstraintDataManager::set_Gu),
          "Jacobian of the inequality constraint")
      .add_property(
          "h",
          bp::make_function(&ConstraintDataManager::get_h,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ConstraintDataManager::set_h),
          "Equality constraint residual")
      .add_property(
          "Hx",
          bp::make_function(&ConstraintDataManager::get_Hx,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ConstraintDataManager::set_Hx),
          "Jacobian of the equality constraint")
      .add_property(
          "Hu",
          bp::make_function(&ConstraintDataManager::get_Hu,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ConstraintDataManager::set_Hu),
          "Jacobian of the equality constraint")
      .def(CopyableVisitor<ConstraintDataManager>());
}

}  // namespace python
}  // namespace crocoddyl
