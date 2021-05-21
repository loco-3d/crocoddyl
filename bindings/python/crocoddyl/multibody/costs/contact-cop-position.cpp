///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Duisburg-Essen, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/contact-cop-position.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactCoPPosition() {  // TODO: Remove once the deprecated update call has been removed in a future
                                       // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelContactCoPPosition> >();

  bp::class_<CostModelContactCoPPosition, bp::bases<CostModelResidual> >(
      "CostModelContactCoPPosition",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameCoPSupport,
               std::size_t>(bp::args("self", "state", "activation", "cop_support", "nu"),
                            "Initialize the contact CoP position cost model.\n\n"
                            ":param state: state of the multibody system\n"
                            ":param activation: activation model (default ActivationModelQuadraticBarrier)\n"
                            ":param cop_support: contact frame Id and cop support region\n"
                            ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameCoPSupport>(
          bp::args("self", "state", "activation", "cop_support"),
          "Initialize the contact CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model (default ActivationModelQuadraticBarrier)\n"
          ":param cop_support: contact frame Id and cop support region"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameCoPSupport, std::size_t>(
          bp::args("self", "state", "cop_support", "nu"),
          "Initialize the contact CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param cop_support: contact frame Id and cop support region\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameCoPSupport>(
          bp::args("self", "state", "cop_support"),
          "Initialize the contact CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param cop_support: contact frame ID and cop support region"))
      .add_property("reference", &CostModelContactCoPPosition::get_reference<FrameCoPSupport>,
                    &CostModelContactCoPPosition::set_reference<FrameCoPSupport>, "reference foot geometry");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
