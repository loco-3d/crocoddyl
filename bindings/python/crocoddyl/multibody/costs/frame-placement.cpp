///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostFramePlacement() {  // TODO: Remove once the deprecated update call has been removed in a future
                                   // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelFramePlacement> >();

  bp::class_<CostModelFramePlacement, bp::bases<CostModelResidual> >(
      "CostModelFramePlacement",
      "This cost function defines a residual vector as r = p - pref, with p and pref as the current and reference "
      "frame placements, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FramePlacement, int>(
          bp::args("self", "state", "activation", "Mref", "nu"),
          "Initialize the frame placement cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Mref: reference frame placement\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FramePlacement>(
          bp::args("self", "state", "activation", "Mref"),
          "Initialize the frame placement cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Mref: reference frame placement"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FramePlacement, int>(
          bp::args("self", "state", "Mref", "nu"),
          "Initialize the frame placement cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FramePlacement>(
          bp::args("self", "state", "Mref"),
          "Initialize the frame placement cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement"))
      .add_property("reference", &CostModelFramePlacement::get_reference<FramePlacement>,
                    &CostModelFramePlacement::set_reference<FramePlacement>, "reference frame placement")
      .add_property("Mref",
                    bp::make_function(&CostModelFramePlacement::get_reference<FramePlacement>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelFramePlacement::set_reference<FramePlacement>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference frame placement");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
