///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/force-base.hpp"

#include "python/crocoddyl/multibody/contact-base.hpp"
#include "python/crocoddyl/multibody/impulse-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeForceAbstract() {
  bp::register_ptr_to_python<std::shared_ptr<ForceDataAbstract> >();

  bp::class_<ForceDataAbstract>(
      "ForceDataAbstract", "Abstract class for force datas.\n\n",
      bp::init<ContactModelAbstract*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create common data shared between force models.\n\n"
          ":param model: force/impulse model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .def(bp::init<ImpulseModelAbstract*,
                    pinocchio::Data*>()[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ForceDataAbstract::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .def_readwrite("frame", &ForceDataAbstract::frame,
                     "frame id of the contact")
      .def_readwrite("type", &ForceDataAbstract::type, "type of contact")
      .add_property(
          "jMf",
          bp::make_getter(&ForceDataAbstract::jMf,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ForceDataAbstract::jMf),
          "local frame placement of the contact frame")
      .add_property("Jc",
                    bp::make_getter(&ForceDataAbstract::Jc,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ForceDataAbstract::Jc), "contact Jacobian")
      .def_readwrite(
          "f", &ForceDataAbstract::f,
          "contact force expressed in the coordinate defined by type")
      .def_readwrite("fext", &ForceDataAbstract::fext,
                     "external spatial force at the parent joint level. Note "
                     "that we could compute the force at the "
                     "contact frame by using jMf (i.e. data.jMf.actInv(data.f)")
      .add_property("df_dx",
                    bp::make_getter(&ForceDataAbstract::df_dx,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ForceDataAbstract::df_dx),
                    "Jacobian of the contact forces expressed in the "
                    "coordinate defined by type")
      .add_property("df_du",
                    bp::make_getter(&ForceDataAbstract::df_du,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ForceDataAbstract::df_du),
                    "Jacobian of the contact forces expressed in the "
                    "coordinate defined by type")
      .def(CopyableVisitor<ForceDataAbstract>());
}

}  // namespace python
}  // namespace crocoddyl
