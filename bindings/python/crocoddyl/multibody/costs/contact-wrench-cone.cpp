///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/contact-wrench-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactWrenchCone() {
  bp::class_<CostModelContactWrenchCone, bp::bases<CostModelAbstract> >(
      "CostModelContactWrenchCone",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameWrenchCone, int>(
          bp::args("self", "state", "activation", "fref", "nu"),
          "Initialize the contact wrench cone cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: frame wrench cone\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameWrenchCone>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the contact wrench cone cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: frame wrench cone"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameWrenchCone, int>(
          bp::args("self", "state", "fref", "nu"),
          "Initialize the contact wrench cone cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param fref: frame wrench cone\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameWrenchCone>(
          bp::args("self", "state", "fref"),
          "Initialize the contact wrench cone cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param fref: frame wrench cone"))
      .def<void (CostModelContactWrenchCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelContactWrenchCone::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact wrench cone cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelContactWrenchCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelContactWrenchCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelContactWrenchCone::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the contact wrench cone cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelContactWrenchCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelContactWrenchCone::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact wrench cone cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelContactWrenchCone::get_reference<FrameWrenchCone>,
                    &CostModelContactWrenchCone::set_reference<FrameWrenchCone>, "reference wrench cone and index")
      .add_property("reference_cone", &CostModelContactWrenchCone::get_reference<WrenchCone>,
                    &CostModelContactWrenchCone::set_reference<WrenchCone>, "reference wrench cone")
      .add_property("reference_id", &CostModelContactWrenchCone::get_reference<FrameIndex>,
                    &CostModelContactWrenchCone::set_reference<FrameIndex>, "reference index");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataContactWrenchCone> >();

  bp::class_<CostDataContactWrenchCone, bp::bases<CostDataAbstract> >(
      "CostDataContactWrenchCone", "Data for contact wrench cone cost.\n\n",
      bp::init<CostModelContactWrenchCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact wrench cone cost data.\n\n"
          ":param model: contact wrench cone cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "contact",
          bp::make_getter(&CostDataContactWrenchCone::contact, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&CostDataContactWrenchCone::contact), "contact data associated with the current cost")
      .add_property("Arr_Rx", bp::make_getter(&CostDataContactWrenchCone::Arr_Rx, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataContactWrenchCone::Arr_Rx),
                    "Intermediate product of Arr (2nd deriv of Activation) with Rx (deriv of residue)")
      .add_property("Arr_Ru", bp::make_getter(&CostDataContactWrenchCone::Arr_Ru, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataContactWrenchCone::Arr_Ru),
                    "Intermediate product of Arr (2nd deriv of Activation) with Ru (deriv of residue)");
}

}  // namespace python
}  // namespace crocoddyl
