///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/contact-force.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactForce() {  // TODO: Remove once the deprecated update call has been removed in a future
                                 // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelContactForce> >();

  bp::class_<CostModelContactForce, bp::bases<CostModelResidual> >(
      "CostModelContactForce",
      "This cost function defines a residual vector as r = f-fref, where f,fref describe the current and reference "
      "the spatial forces, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameForce, int>(
          bp::args("self", "state", "activation", "fref", "nu"),
          "Initialize the contact force cost model.\n\n"
          "Note that the activation.nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameForce>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the contact force cost model.\n\n"
          "The default nu is obtained from state.nv.\n"
          "Note that the activation.nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce, int, int>(
          bp::args("self", "state", "fref", "nr", "nu"),
          "Initialize the contact force cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          "Note that the nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"
          ":param nr: dimension of force vector (>= 6)\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce, int>(
          bp::args("self", "state", "fref", "nr"),
          "Initialize the contact force cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          "Note that the nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"
          ":param nr: dimension of force vector (>= 6)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce>(
          bp::args("self", "state", "fref"),
          "Initialize the contact force cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact force in the contact coordinates"))
      .add_property("reference", &CostModelContactForce::get_reference<FrameForce>,
                    &CostModelContactForce::set_reference<FrameForce>,
                    "reference spatial contact force in the contact coordinates")
      .add_property("fref",
                    bp::make_function(&CostModelContactForce::get_reference<FrameForce>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelContactForce::set_reference<FrameForce>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference spatial contact force in the contact coordinates");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
