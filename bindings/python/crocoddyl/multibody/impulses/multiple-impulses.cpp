///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/map-converter.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ImpulseModelMultiple_addImpulse_wrap,
                                       ImpulseModelMultiple::addImpulse, 2, 3)

void exposeImpulseMultiple() {
  // Register custom converters between std::map and Python dict
  typedef std::shared_ptr<ImpulseItem> ImpulseItemPtr;
  typedef std::shared_ptr<ImpulseDataAbstract> ImpulseDataPtr;
  StdMapPythonVisitor<
      std::string, ImpulseItemPtr, std::less<std::string>,
      std::allocator<std::pair<const std::string, ImpulseItemPtr>>,
      true>::expose("StdMap_ImpulseItem");
  StdMapPythonVisitor<
      std::string, ImpulseDataPtr, std::less<std::string>,
      std::allocator<std::pair<const std::string, ImpulseDataPtr>>,
      true>::expose("StdMap_ImpulseData");

  bp::register_ptr_to_python<std::shared_ptr<ImpulseItem>>();

  bp::class_<ImpulseItem>(
      "ImpulseItem", "Describe a impulse item.\n\n",
      bp::init<std::string, std::shared_ptr<ImpulseModelAbstract>,
               bp::optional<bool>>(
          bp::args("self", "name", "impulse", "active"),
          "Initialize the impulse item.\n\n"
          ":param name: impulse name\n"
          ":param impulse: impulse model\n"
          ":param active: True if the impulse is activated (default true)"))
      .def_readwrite("name", &ImpulseItem::name, "impulse name")
      .add_property(
          "impulse",
          bp::make_getter(&ImpulseItem::impulse,
                          bp::return_value_policy<bp::return_by_value>()),
          "impulse model")
      .def_readwrite("active", &ImpulseItem::active, "impulse status")
      .def(CopyableVisitor<ImpulseItem>())
      .def(PrintableVisitor<ImpulseItem>());
  ;

  bp::register_ptr_to_python<std::shared_ptr<ImpulseModelMultiple>>();

  bp::class_<ImpulseModelMultiple>(
      "ImpulseModelMultiple",
      bp::init<std::shared_ptr<StateMultibody>>(
          bp::args("self", "state"),
          "Initialize the multiple impulse model.\n\n"
          ":param state: state of the multibody system"))
      .def(
          "addImpulse", &ImpulseModelMultiple::addImpulse,
          ImpulseModelMultiple_addImpulse_wrap(
              bp::args("self", "name", "impulse", "active"),
              "Add an impulse item.\n\n"
              ":param name: impulse name\n"
              ":param impulse: impulse model\n"
              ":param active: True if the impulse is activated (default true)"))
      .def("removeImpulse", &ImpulseModelMultiple::removeImpulse,
           bp::args("self", "name"),
           "Remove an impulse item.\n\n"
           ":param name: impulse name")
      .def("changeImpulseStatus", &ImpulseModelMultiple::changeImpulseStatus,
           bp::args("self", "name", "active"),
           "Change the impulse status.\n\n"
           ":param name: impulse name\n"
           ":param active: impulse status (true for active and false for "
           "inactive)")
      .def("calc", &ImpulseModelMultiple::calc, bp::args("self", "data", "x"),
           "Compute the impulse Jacobian and drift.\n\n"
           "The rigid impulse model throught acceleration-base holonomic "
           "constraint\n"
           "of the impulse frame placement.\n"
           ":param data: impulse data\n"
           ":param x: state point (dim. state.nx)")
      .def("calcDiff", &ImpulseModelMultiple::calcDiff,
           bp::args("self", "data", "x"),
           "Compute the derivatives of the impulse holonomic constraint.\n\n"
           "The rigid impulse model throught acceleration-base holonomic "
           "constraint\n"
           "of the impulse frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: impulse data\n"
           ":param x: state point (dim. state.nx)")
      .def("updateVelocity", &ImpulseModelMultiple::updateVelocity,
           bp::args("self", "data", "vnext"),
           "Update the system velocity after impulse.\n\n"
           ":param data: impulse data\n"
           ":param vnext: velocity after impulse (dimension nv)")
      .def("updateForce", &ImpulseModelMultiple::updateForce,
           bp::args("self", "data", "force"),
           "Update the spatial impulse defined in frame coordinate.\n\n"
           ":param data: impulse data\n"
           ":param force: force vector (dimension ni)")
      .def("updateVelocityDiff", &ImpulseModelMultiple::updateVelocityDiff,
           bp::args("self", "data", "dvnext_dx"),
           "Update the Jacobian of the system velocity after impulse.\n\n"
           ":param data: impulse data\n"
           ":param dvnext_dx: Jacobian of the impulse velocity (dimension "
           "nv*ndx)")
      .def("updateForceDiff", &ImpulseModelMultiple::updateForceDiff,
           bp::args("self", "data", "df_dx"),
           "Update the Jacobian of the spatial impulse defined in frame "
           "coordinate.\n\n"
           ":param data: impulse data\n"
           ":param df_dx: Jacobian of the impulse force (dimension ni*ndx)")
      .def("updateRneaDiff", &ImpulseModelMultiple::updateRneaDiff,
           bp::args("self", "data", "pinocchio"),
           "Update the RNEA derivative dtau_dq by by adding the skew term "
           "(necessary for impulses expressed in\n"
           "LOCAL_WORLD_ALIGNED / WORLD).\n\n"
           ":param data: impulse data\n"
           ":param pinocchio: Pinocchio data")
      .def("createData", &ImpulseModelMultiple::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the total impulse data.\n\n"
           ":param data: Pinocchio data\n"
           ":return total impulse data.")
      .add_property(
          "impulses",
          bp::make_function(&ImpulseModelMultiple::get_impulses,
                            bp::return_value_policy<bp::return_by_value>()),
          "stack of impulses")
      .add_property(
          "state",
          bp::make_function(&ImpulseModelMultiple::get_state,
                            bp::return_value_policy<bp::return_by_value>()),
          "state of the multibody system")
      .add_property("nc", bp::make_function(&ImpulseModelMultiple::get_nc),
                    "dimension of the active impulse vector")
      .add_property("nc_total",
                    bp::make_function(&ImpulseModelMultiple::get_nc_total),
                    "dimension of the total impulse vector")
      .add_property(
          "active_set",
          bp::make_function(&ImpulseModelMultiple::get_active_set,
                            bp::return_value_policy<bp::return_by_value>()),
          "set of names of active impulse items")
      .add_property(
          "inactive_set",
          bp::make_function(&ImpulseModelMultiple::get_inactive_set,
                            bp::return_value_policy<bp::return_by_value>()),
          "set of names of inactive impulse items")
      .def("getImpulseStatus", &ImpulseModelMultiple::getImpulseStatus,
           bp::args("self", "name"),
           "Return the impulse status of a given impulse name.\n\n"
           ":param name: impulse name")
      .def(CopyableVisitor<ImpulseModelMultiple>())
      .def(PrintableVisitor<ImpulseModelMultiple>());

  bp::register_ptr_to_python<std::shared_ptr<ImpulseDataMultiple>>();

  bp::class_<ImpulseDataMultiple>(
      "ImpulseDataMultiple", "Data class for multiple impulses.\n\n",
      bp::init<ImpulseModelMultiple*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create multi-impulse data.\n\n"
          ":param model: multi-impulse model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .add_property("Jc",
                    bp::make_getter(&ImpulseDataMultiple::Jc,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ImpulseDataMultiple::Jc),
                    "Jacobian for all impulses (active and inactive)")
      .add_property(
          "dv0_dq",
          bp::make_getter(&ImpulseDataMultiple::dv0_dq,
                          bp::return_internal_reference<>()),
          bp::make_setter(&ImpulseDataMultiple::dv0_dq),
          "Jacobian of the previous impulse velocity (active and inactive)")
      .add_property("vnext",
                    bp::make_getter(&ImpulseDataMultiple::vnext,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ImpulseDataMultiple::vnext),
                    "impulse system velocity")
      .add_property("dvnext_dx",
                    bp::make_getter(&ImpulseDataMultiple::dvnext_dx,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ImpulseDataMultiple::dvnext_dx),
                    "Jacobian of the impulse system velocity")
      .add_property(
          "impulses",
          bp::make_getter(&ImpulseDataMultiple::impulses,
                          bp::return_value_policy<bp::return_by_value>()),
          "stack of impulses data")
      .def_readwrite("fext", &ImpulseDataMultiple::fext,
                     "external spatial forces")
      .def(CopyableVisitor<ImpulseDataMultiple>());
}

}  // namespace python
}  // namespace crocoddyl
