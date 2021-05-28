///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/force-base.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/contact-base.hpp"
#include "python/crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {
namespace python {

void exposeForceAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<ForceDataAbstract> >();

  bp::class_<ForceDataAbstract, boost::noncopyable>(
      "ForceDataAbstract", "Abstract class for force datas.\n\n",
      bp::init<ContactModelAbstract*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create common data shared between force models.\n\n"
          ":param model: force/impulse model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .def(bp::init<ImpulseModelAbstract*,
                    pinocchio::Data*>()[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio", bp::make_getter(&ForceDataAbstract::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("jMf", bp::make_getter(&ForceDataAbstract::jMf, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ForceDataAbstract::jMf), "local frame placement of the contact frame")
      .add_property("Jc", bp::make_getter(&ForceDataAbstract::Jc, bp::return_internal_reference<>()),
                    bp::make_setter(&ForceDataAbstract::Jc), "contact Jacobian")
      .add_property("df_dx", bp::make_getter(&ForceDataAbstract::df_dx, bp::return_internal_reference<>()),
                    bp::make_setter(&ForceDataAbstract::df_dx), "Jacobian of the contact forces")
      .add_property("df_du", bp::make_getter(&ForceDataAbstract::df_du, bp::return_internal_reference<>()),
                    bp::make_setter(&ForceDataAbstract::df_du), "Jacobian of the contact forces")
      .def_readwrite("frame", &ForceDataAbstract::frame, "frame index of the contact frame")
      .def_readwrite("f", &ForceDataAbstract::f,
                     "external spatial force at the parent joint level. Note that we could compute the force at the "
                     "contact frame by using jMf (i.e. data.jMf.actInv(data.f)");
}

}  // namespace python
}  // namespace crocoddyl
