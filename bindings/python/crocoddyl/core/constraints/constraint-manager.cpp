///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <string>
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/utils/map-converter.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ConstraintModelManager_addConstraint_wrap,
                                       ConstraintModelManager::addConstraint, 2, 3)

void exposeConstraintManager() {
  // Register custom converters between std::map and Python dict
  typedef boost::shared_ptr<ConstraintItem> ConstraintItemPtr;
  typedef boost::shared_ptr<ConstraintDataAbstract> ConstraintDataPtr;
  StdMapPythonVisitor<std::string, ConstraintItemPtr, std::less<std::string>,
                      std::allocator<std::pair<const std::string, ConstraintItemPtr> >,
                      true>::expose("StdMap_ConstraintItem");
  StdMapPythonVisitor<std::string, ConstraintDataPtr, std::less<std::string>,
                      std::allocator<std::pair<const std::string, ConstraintDataPtr> >,
                      true>::expose("StdMap_ConstraintData");

  bp::register_ptr_to_python<boost::shared_ptr<ConstraintItem> >();

  bp::class_<ConstraintItem>("ConstraintItem", "Describe a constraint item.\n\n",
                             bp::init<std::string, boost::shared_ptr<ConstraintModelAbstract>, bp::optional<bool> >(
                                 bp::args("self", "name", "constraint", "active"),
                                 "Initialize the constraint item.\n\n"
                                 ":param name: constraint name\n"
                                 ":param constraint: constraint model\n"
                                 ":param active: True if the constraint is activated (default true)"))
      .def_readwrite("name", &ConstraintItem::name, "constraint name")
      .add_property("constraint",
                    bp::make_getter(&ConstraintItem::constraint, bp::return_value_policy<bp::return_by_value>()),
                    "constraint model")
      .def_readwrite("active", &ConstraintItem::active, "constraint status");
  ;

  bp::register_ptr_to_python<boost::shared_ptr<ConstraintModelManager> >();

  bp::class_<ConstraintModelManager>("ConstraintModelManager", bp::init<boost::shared_ptr<StateAbstract>, std::size_t>(
                                                                   bp::args("self", "state", "nu"),
                                                                   "Initialize the total constraint model.\n\n"
                                                                   ":param state: state description\n"
                                                                   ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the total constraint model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract> >(bp::args("self", "state"),
                                                       "Initialize the total constraint model.\n\n"
                                                       "For this case the default nu is equals to model.nv.\n"
                                                       ":param state: state description"))
      .def("addConstraint", &ConstraintModelManager::addConstraint,
           ConstraintModelManager_addConstraint_wrap(
               bp::args("self", "name", "constraint", "active"),
               "Add a constraint item.\n\n"
               ":param name: constraint name\n"
               ":param constraint: constraint model\n"
               ":param active: True if the constraint is activated (default true)"))
      .def("removeConstraint", &ConstraintModelManager::removeConstraint, bp::args("self", "name"),
           "Remove a constraint item.\n\n"
           ":param name: constraint name")
      .def("changeConstraintStatus", &ConstraintModelManager::changeConstraintStatus,
           bp::args("self", "name", "active"),
           "Change the constraint status.\n\n"
           ":param name: constraint name\n"
           ":param active: constraint status (true for active and false for inactive)")
      .def<void (ConstraintModelManager::*)(const boost::shared_ptr<ConstraintDataManager>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelManager::calc, bp::args("self", "data", "x", "u"),
          "Compute the total constraint.\n\n"
          ":param data: constraint-sum data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ConstraintModelManager::*)(const boost::shared_ptr<ConstraintDataManager>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelManager::calc, bp::args("self", "data", "x"))
      .def<void (ConstraintModelManager::*)(const boost::shared_ptr<ConstraintDataManager>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelManager::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the total constraint.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ConstraintModelManager::*)(const boost::shared_ptr<ConstraintDataManager>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelManager::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ConstraintModelManager::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the total constraint data.\n\n"
           ":param data: shared data\n"
           ":return total constraint data.")
      .add_property(
          "state",
          bp::make_function(&ConstraintModelManager::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state description")
      .add_property(
          "constraints",
          bp::make_function(&ConstraintModelManager::get_constraints, bp::return_value_policy<bp::return_by_value>()),
          "stack of constraints")
      .add_property("nu", bp::make_function(&ConstraintModelManager::get_nu), "dimension of control vector")
      .add_property("ng", bp::make_function(&ConstraintModelManager::get_ng),
                    "number of active inequality constraints")
      .add_property("nh", bp::make_function(&ConstraintModelManager::get_nh), "number of active equality constraints")
      .add_property("ng_total", bp::make_function(&ConstraintModelManager::get_ng_total),
                    "number of the total inequality constraints")
      .add_property("nh_total", bp::make_function(&ConstraintModelManager::get_nh_total),
                    "number of the total equality constraints")
      .add_property(
          "active",
          bp::make_function(&ConstraintModelManager::get_active, bp::return_value_policy<bp::return_by_value>()),
          "name of active constraint items")
      .add_property(
          "inactive",
          bp::make_function(&ConstraintModelManager::get_inactive, bp::return_value_policy<bp::return_by_value>()),
          "name of inactive constraint items")
      .def("getConstraintStatus", &ConstraintModelManager::getConstraintStatus, bp::args("self", "name"),
           "Return the constraint status of a given constraint name.\n\n"
           ":param name: constraint name");

  bp::register_ptr_to_python<boost::shared_ptr<ConstraintDataManager> >();

  bp::class_<ConstraintDataManager>("ConstraintDataManager", "Class for total constraint data.\n\n",
                                    bp::init<ConstraintModelManager*, DataCollectorAbstract*>(
                                        bp::args("self", "model", "data"),
                                        "Create total constraint data.\n\n"
                                        ":param model: total constraint model\n"
                                        ":param data: shared data")[bp::with_custodian_and_ward<1, 3>()])
      .def("shareMemory", &ConstraintDataManager::shareMemory<DifferentialActionDataAbstract>,
           bp::args("self", "model"),
           "Share memory with a given differential action data\n\n"
           ":param model: differential action data that we want to share memory")
      .def("shareMemory", &ConstraintDataManager::shareMemory<ActionDataAbstract>, bp::args("self", "model"),
           "Share memory with a given action data\n\n"
           ":param model: action data that we want to share memory")
      .add_property(
          "constraints",
          bp::make_getter(&ConstraintDataManager::constraints, bp::return_value_policy<bp::return_by_value>()),
          "stack of constraints data")
      .add_property("shared", bp::make_getter(&ConstraintDataManager::shared, bp::return_internal_reference<>()),
                    "shared data")
      .add_property("g",
                    bp::make_function(&ConstraintDataManager::get_g, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ConstraintDataManager::set_g), "Inequality constraint residual")
      .add_property("Gx",
                    bp::make_function(&ConstraintDataManager::get_Gx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ConstraintDataManager::set_Gx), "Jacobian of the inequality constraint")
      .add_property("Gu",
                    bp::make_function(&ConstraintDataManager::get_Gu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ConstraintDataManager::set_Gu), "Jacobian of the inequality constraint")
      .add_property("h",
                    bp::make_function(&ConstraintDataManager::get_h, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ConstraintDataManager::set_h), "Equality constraint residual")
      .add_property("Hx",
                    bp::make_function(&ConstraintDataManager::get_Hx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ConstraintDataManager::set_Hx), "Jacobian of the equality constraint")
      .add_property("Hu",
                    bp::make_function(&ConstraintDataManager::get_Hu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ConstraintDataManager::set_Hu), "Jacobian of the equality constraint");
}

}  // namespace python
}  // namespace crocoddyl
